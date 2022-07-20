import json
from dataclasses import dataclass
from collections import defaultdict


class Item():
    def __init__(self, name):
        self.name = name
        self.children = []
        self.index_in_parent = None
        self.parent = None
        self.softmax_start_index = None

    def add_child(self, child):
        assert child.index_in_parent is None
        assert child.parent is None
        child.index_in_parent = len(self.children)
        child.parent = self
        self.children.append(child)
        return child

    def set_softmax_start_index(self, current_index):
        assert self.softmax_start_index is None
        self.softmax_start_index = current_index
        current_index += len(self.children)
        self.softmax_end_index = current_index
        current_index += 
        current_index


class Image(Item):
    def __init__(self, image_id, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_id = image_id
        self.path = path


class Category(Item):
    def __init__(self, category_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category_id = category_id


class Family(Item):
    pass


class Order(Item):
    pass


class MetadataManager(Item):
    def __init__(self, train_dir):
        super().__init__("metadata")

        self.train_dir = train_dir
        with open(train_dir/"metadata.json", 'r') as f:
            metadata = json.load(f)

        print("Getting Hierarchies")
        self.category_to_order = {}
        self.category_to_family = {}
        self.get_category = {}
        self.get_image = {}

        print("Categories")
        for category_dict in metadata['categories']:
            category_name = category_dict["order"]
            order = self.get_child_by_name(category_name)
            if not order:
                order = Order(name=category_name)
                self.add_child(order)

            family_name = category_dict["family"]
            family = self.get_child_by_name(family_name)
            if not family:
                family = Family(name=family_name)
                order.add_child(family)

            category_name = category_dict["family"]
            category_id = category_dict['id']
            category = Category(name=category_name, category_id=category_id)
            self.get_category[category_id] = category

        print("Reading Images")
        for image_dict in metadata["images"]:
            image_id = image_dict['id']
            path = image_dict['file_name']
            image = Image(image_id=image_id, path=path)
            self.get_image[image_id] = image

        print("Reading Annotations")
        for annotation in metadata['annotations']:
            image_id = annotation['image_id']
            image = self.get_image[image_id]

            category = self.get_category[annotation['category_id']]
            category.add_child(image)

        self.category_id_to_family_index = {}
        current_index = len(self.orders)
        self.family_start = {}
        self.family_end = {}
        for order in self.children:
            self.family_start[order.index_in_parent] = current_index
            current_index += len(order.children)
            self.family_end[order.index_in_parent] = current_index

            for family in order.children:
                self.category_start[family.index_in_parent] = current_index
                current_index += len(family.children)
                self.category_end[family.index_in_parent] = current_index

    def get_path(self, image:Image):
        return self.train_dir/image.path

    def get_order(self, image:Image):
        return image.parent.parent.parent.index_in_parent

    def order_count(self):
        return len(self.children)

    

        # family_start,
        # family_end,
        # family_index,
        # category_start,
        # category_end,
        # category_index,
