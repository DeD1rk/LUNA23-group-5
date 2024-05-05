from pathlib import Path

import torch
from torch.nn.functional import binary_cross_entropy, cross_entropy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from .constants import NODULETYPE_MAPPING
from .dataset import LUNADataset, get_balancing_weights
from .model import Model
from .utils import worker_init_fn


def dice_loss(input, target):
    """Function to compute dice loss
    source: https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398

    Args:
        input (torch.Tensor): predictions
        target (torch.Tensor): ground truth mask

    Returns:
        dice loss: 1 - dice coefficient
    """
    smooth = 1.0

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class Trainer:
    def __init__(
        self,
        data_dir: Path,
        fold: int = 0,
        epochs: int = 100,
        batch_size: int = 8,
    ):
        self.data_dir = data_dir
        self.fold = fold
        self.epochs = epochs
        self.batch_size = batch_size

        torch.backends.cudnn.benchmark = True
        self.device = torch.device("cuda:0")

        # For testing locally when no GPU is available:
        # self.device = torch.device("cpu:0")

        self.model = Model().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
        )

        dataset_train = LUNADataset(self.data_dir, fold=self.fold)
        dataset_valid = LUNADataset(self.data_dir, fold=self.fold, validation=True)

        sampler = WeightedRandomSampler(
            torch.DoubleTensor(
                get_balancing_weights(dataset_train.dataframe.malignancy.values)
                * get_balancing_weights(
                    [
                        NODULETYPE_MAPPING[t]
                        for t in dataset_train.dataframe.noduletype.values
                    ]
                )
            ),
            len(dataset_train),
        )

        self.dataloader_train = DataLoader(
            dataset=dataset_train,
            batch_size=self.batch_size,
            sampler=sampler,
            worker_init_fn=worker_init_fn,
        )

        self.dataloader_valid = DataLoader(
            dataset=dataset_valid,
            batch_size=self.batch_size,
            worker_init_fn=worker_init_fn,
        )

    def call_model(self, batch: dict):
        images = batch["image"].to(self.device)

        labels = {
            "segmentation": batch["segmentation_label"].to(self.device),
            "noduletype": batch["noduletype_label"].to(self.device),
            "malignancy": batch["malignancy_label"].to(self.device),
        }

        outputs = self.model(images)

        # TODO: numpyify and move to cpu whatever is needed

        losses = {
            "segmentation": dice_loss(outputs["segmentation"], labels["segmentation"]),
            "noduletype": cross_entropy(outputs["noduletype"], labels["noduletype"]),
            "malignancy": binary_cross_entropy(
                outputs["malignancy"], labels["malignancy"].float()
            ),
        }

        losses["total"] = sum(losses.values())

        return outputs, labels, losses

    def train_epoch(self):
        self.model.train()

        # TODO: logging, tqdm and keeping metrics
        for batch in self.dataloader_train:
            self.optimizer.zero_grad()
            batch_predictions, batch_labels, batch_losses = self.call_model(batch)

            batch_losses["total"].backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        # TODO: return metrics

    def validation(self):
        self.model.eval()
        losses = {
            "total": 0,
            "segmentation": 0,
            "noduletype": 0,
            "malignancy": 0,
        }

        # TODO: logging, tqdm and keeping metrics
        with torch.no_grad():
            for batch in self.dataloader_valid:
                batch_predictions, batch_labels, batch_losses = self.call_model(batch)
                for loss, value in batch_losses.items():
                    losses[loss] += value

    def train(self):
        # TODO: logging, keeping metrics and saving best model.
        for epoch in range(self.epochs):
            self.train_epoch()
            self.validation()
