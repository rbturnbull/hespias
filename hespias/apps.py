from functools import partial
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
from .metadata import MetadataManager
from hierarchicalsoftmax import HierarchicalSoftmaxLoss
from hierarchicalsoftmax import metrics


class DictionaryGetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, key):
        value = self.dictionary[key]
        return value


class DictionaryPathGetter:
    def __init__(self, dictionary, base_path):
        self.dictionary = dictionary
        self.base_path = base_path

    def __call__(self, key):
        value =  self.base_path/self.dictionary[key]
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
        width: int = fa.Param(default=224, help="The width to resize all the images to."),
        height: int = fa.Param(default=224, help="The height to resize all the images to."),
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
        
        self.metadata = MetadataManager(train_dir)
        image_ids = self.metadata.image_ids()

        if max_images and len(image_ids) >= max_images:
            image_ids = image_ids[:max_images]

        print("Building datablock")
        datablock = DataBlock(
            blocks=[ImageBlock, CategoryBlock],
            get_x=DictionaryGetter(self.metadata.get_image),
            get_y=DictionaryGetter(self.metadata.image_id_to_node_id),
            splitter=RandomSplitter(validation_proportion),
            item_tfms=RandomResizedCrop((height, width)),
        )

        print("Building dataloaders")
        dataloaders = datablock.dataloaders(image_ids, bs=batch_size)
        dataloaders.c = self.metadata.root.layer_size

        print("finished building dataloaders")

        return dataloaders

    def get_loss(self):
        return HierarchicalSoftmaxLoss(root=self.metadata.root)

    def metrics(self):
        return [partial(metrics.greedy_accuracy, root=self.metadata.root), partial(metrics.greedy_f1_score, root=self.metadata.root)]

    def monitor(self):
        return "greedy_f1_score"
