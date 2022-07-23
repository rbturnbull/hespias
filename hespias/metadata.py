import json
from hierarchicalsoftmax import SoftmaxNode

class MetadataManager():
    def __init__(self, train_dir, metadata_filename="metadata.json"):
        super().__init__()

        self.train_dir = train_dir
        with open(train_dir/metadata_filename, 'r') as f:
            metadata = json.load(f)

        print("Getting Hierarchies")
        self.get_category = {}
        self.get_image = {}
        self.image_id_to_node_id = {}
        self.root = SoftmaxNode(name="root")

        print("Categories")
        for category_dict in metadata['categories']:
            category_name = category_dict["order"]
            order = self.root.get_child_by_name(category_name)
            if not order:
                order = SoftmaxNode(name=category_name, type="order", parent=self.root)

            family_name = category_dict["family"]
            family = order.get_child_by_name(family_name)
            if not family:
                family = SoftmaxNode(name=family_name, type="family", parent=order)

            category_name = category_dict["family"]
            category_id = category_dict['id']
            category = SoftmaxNode(name=category_name, category_id=category_id, type="category", parent=family)
            self.get_category[category_id] = category

        self.root.set_indexes()

        print("Reading Images")
        for image_dict in metadata["images"]:
            image_id = image_dict['id']
            self.get_image[image_id] = self.train_dir/image_dict['file_name']

        print("Reading Annotations")
        for annotation in metadata['annotations']:
            image_id = annotation['image_id']

            category = self.get_category[annotation['category_id']]
            node_id = self.root.node_to_id[category]
            self.image_id_to_node_id[image_id] = node_id

        print("Determining Mean Aspect Ratio")
        self.mean_aspect_ratio = 0.0
        for image in metadata['images']:
            small_side = min(image['width'], image['height'])
            long_side = max(image['width'], image['height'])
            self.mean_aspect_ratio += small_side/long_side
        self.mean_aspect_ratio /= len(metadata['images'])


    def image_ids(self):
        return list(self.get_image.keys())

