import json
from pathlib import Path
from fastai.data.core import DataLoaders
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import RandomSplitter
from fastai.vision.data import ImageBlock
from fastai.vision.augment import RandomResizedCrop

from rich.console import Console
console = Console()

import fastapp as fa
from fastapp.vision import VisionApp


class DictionaryGetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, key):
        return self.dictionary[key]


class Hespias(VisionApp):
    """
    A herbarium specimen classifier.
    """
    def dataloaders(
        self,
        train_dir:Path = fa.Param(help="The training directory that's part of the herbarium-2021-fgvc8 competition on Kaggle."), 
        batch_size:int = fa.Param(default=32, help="The batch size."),
        validation_proportion: float = fa.Param(
            default=0.2,
            help="The proportion of the dataset to keep for validation. Used if `validation_column` is not in the dataset.",
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

        category_to_order = {}
        category_to_family = {}

        for category_dict in metadata['categories']:
            category_to_order[category_dict['id']] = category_dict["order"]
            category_to_family[category_dict['id']] = category_dict["family"]
        
        image_id_to_path = {}
        for image_dict in metadata["images"]:
            image_id_to_path[image_dict['id']] = train_dir/image_dict['file_name']

        image_id_to_order = {}
        image_id_to_family = {}
        image_id_to_category = {}
        for annotation in metadata['annotations']:
            image_id = annotation['image_id']
            category = annotation['category_id']

            assert image_id in image_id_to_path
            image_id_to_order[image_id] = category_to_order[category]
            image_id_to_family[image_id] = category_to_family[category]
            image_id_to_category[image_id] = category
        
        datablock = DataBlock(
            blocks=[ImageBlock, CategoryBlock],
            get_x=DictionaryGetter(image_id_to_path),
            get_y=DictionaryGetter(image_id_to_order),
            splitter=RandomSplitter(validation_proportion),
            item_tfms=RandomResizedCrop(256, min_scale=0.35),
        )

        return datablock.dataloaders(image_id_to_path.keys(), bs=batch_size)

