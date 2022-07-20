from torch import nn
from torch import Tensor
import torch
import torch.nn.functional as F

from fastai.torch_core import default_device


class HespiasLoss(nn.Module):
    def __init__(
        self,
        order_count,
        category_to_order_id,
        category_to_family_id,
        family_start,
        family_end,
        family_index,
        category_start,
        category_end,
        category_index,
        order_label_smoothing=0.0,
        family_label_smoothing=0.0,
        category_label_smoothing=0.0,
        order_lambda=1.0,
        family_lambda=1.0,
        **kwargs
    ):
        super().__init__(self, metadata, **kwargs)
        self.metadata = metadata
        self.order_count = order_count
        self.category_to_order_id = category_to_order_id
        self.category_to_family_id = category_to_family_id

        self.order_lambda = order_lambda
        self.family_lambda = family_lambda

        self.family_start = family_start
        self.family_end = family_end
        self.family_index = family_index
        
        self.category_start = category_start
        self.category_end = category_end
        self.category_index = category_index
        
        self.order_label_smoothing = order_label_smoothing
        self.family_label_smoothing = family_label_smoothing
        self.category_label_smoothing = category_label_smoothing

    def forward(self, batch_predictions: Tensor, target_categories: Tensor) -> Tensor:
        categories = (self.metadata.get_category[category_id] for category_id in target_categories)
        orders = torch.as_tensor( (category.parent.parent.index_in_parent for category in categories) )
        order_loss = F.cross_entropy(
            batch_predictions[:, :self.metadata.order_count()],
            orders,
            label_smoothing=self.order_label_smoothing,
        )

        family_loss = 0.0
        category_loss = 0.0
        for prediction, category, order in zip(batch_predictions, categories, orders):
            start = self.metadata.family_start[order.index_in_parent]
            end = self.metadata.family_end[order.index_in_parent]
            family = category.parent
            family_index = family.index_in_parent

            family_loss += F.cross_entropy(
                prediction[start:end],
                family_index,
                label_smoothing=self.family_label_smoothing,
            )

            start = self.metadata.category_start[family_index]
            end = self.metadata.category_end[family_index]
            family = category.parent

            start = self.category_start[family_id]
            end = self.category_end[family_id]
            category_index = self.category_index[category_id]

            category_loss += F.cross_entropy(
                prediction[start:end],
                category_index,
                label_smoothing=self.family_label_smoothing,
            )

        # divide by batch size # TODO weight?
        batch_size = len(batch_predictions)
        family_loss /= batch_size
        category_loss /= batch_size

        return category_loss + self.family_lambda * family_loss + self.order_lambda * order_loss
