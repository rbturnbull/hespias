import json
from pathlib import Path
from fastai.data.core import DataLoaders
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import RandomSplitter
from fastai.vision.data import ImageBlock
from fastai.vision.augment import RandomResizedCrop, Resize

from rich.console import Console
console = Console()

import fastapp as fa
from fastapp.vision import VisionApp


class DictionaryGetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, key):
        value = self.dictionary[key]
        print(key, value)
        return value


class DictionaryPathGetter:
    def __init__(self, dictionary, base_path):
        self.dictionary = dictionary
        self.base_path = base_path

    def __call__(self, key):
        value =  self.base_path/self.dictionary[key]
        print(key, value)
        return value


class Hespias(VisionApp):
    """
    A herbarium specimen classifier.
    """
    def dataloaders(
        self,
        train_dir:Path = fa.Param(help="The training directory that's part of the herbarium-2021-fgvc8 competition on Kaggle."), 
        batch_size:int = fa.Param(default=16, help="The batch size."),
        validation_proportion: float = fa.Param(
            default=0.2,
            help="The proportion of the dataset to keep for validation. Used if `validation_column` is not in the dataset.",
        ),
        max_images:int = fa.Param(
            default=None,
            help="The maximum number of images to use for training and validation (if set).",
        ),
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Hespias uses in training and prediction.

        Args:
            train_dir (Path): The training directory that's part of the herbarium-2021-fgvc8 competition on Kaggle.
            batch_size (int, optional): The number of elements to use in a batch for training and prediction. Defaults to 32.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        train_dir = Path(train_dir)
        with open(train_dir/"metadata.json", 'r') as f:
            metadata = json.load(f)

        print("Getting Hierarchies")
        category_to_order = {}
        category_to_family = {}

        for category_dict in metadata['categories']:
            category_to_order[category_dict['id']] = category_dict["order"]
            category_to_family[category_dict['id']] = category_dict["family"]
        
        print("Getting Image Paths")
        print('max_images', max_images)
        image_id_to_path = {}
        for image_dict in metadata["images"]:
            image_id_to_path[image_dict['id']] = image_dict['file_name']

            if max_images and len(image_id_to_path) >= max_images:
                break

        print("image ids:")
        image_ids = list(image_id_to_path.keys())
        print(image_ids)

        print("Getting Ys")
        image_id_to_order = {}
        image_id_to_family = {}
        image_id_to_category = {}
        for annotation in metadata['annotations']:
            image_id = annotation['image_id']
            if image_id not in image_ids:
                continue
            category = annotation['category_id']
            image_id_to_order[image_id] = category_to_order[category]
            image_id_to_family[image_id] = category_to_family[category]
            image_id_to_category[image_id] = category
        
        print("Building datablock")
        datablock = DataBlock(
            blocks=[ImageBlock, CategoryBlock],
            get_x=DictionaryPathGetter(image_id_to_path, train_dir),
            get_y=DictionaryGetter(image_id_to_order),
            splitter=RandomSplitter(validation_proportion),
            # item_tfms=Resize(256),
        )

        print("Building dataloaders")
        dataloaders = datablock.dataloaders(image_ids, bs=batch_size)

        print("finished building dataloaders")

        return dataloaders

