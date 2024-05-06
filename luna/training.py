from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas
import sklearn.metrics as skl_metrics
import torch
from torch.nn.functional import binary_cross_entropy, cross_entropy
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

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
        save_dir: Path,
        fold: int = 0,
        epochs: int = 100,
        batch_size: int = 8,
        task_weights: dict = {
            "segmentation": 1.0,
            "noduletype": 1.0,
            "malignancy": 1.0,
        },
    ):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.fold = fold
        self.epochs = epochs
        self.batch_size = batch_size
        self.task_weights = task_weights.copy()

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

        losses = {
            "segmentation": dice_loss(outputs["segmentation"], labels["segmentation"]),
            "noduletype": cross_entropy(outputs["noduletype"], labels["noduletype"]),
            "malignancy": binary_cross_entropy(
                outputs["malignancy"], labels["malignancy"].float()
            ),
        }

        losses["total"] = sum([self.task_weights[task] * loss for task, loss in losses.items()])

        outputs = {
            "segmentation": outputs["segmentation"].detach().cpu().numpy(),
            "noduletype": outputs["noduletype"].detach().cpu().numpy(),
            "malignancy": outputs["malignancy"].detach().cpu().numpy(),
        }

        labels = {task: batch_labels.cpu() for task, batch_labels in labels.items()}

        return outputs, labels, losses

    def train_epoch(self):
        self.model.train()
        losses = defaultdict(list)
        predictions = defaultdict(list)
        labels = defaultdict(list)

        for batch in tqdm(self.dataloader_train, desc="Training"):
            self.optimizer.zero_grad()
            batch_predictions, batch_labels, batch_loss = self.call_model(batch)

            # Store the predictions, labels and losses to later aggregate them.
            for loss, value in batch_loss.items():
                losses[loss].append(value.item())
            for task, value in batch_predictions.items():
                predictions[task].extend(value)
            for task, value in batch_labels.items():
                labels[task].extend(value)

            batch_loss["total"].backward()
            self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "loss_segmentation": np.mean(losses["segmentation"]),
            "loss_noduletype": np.mean(losses["noduletype"]),
            "loss_malignancy": np.mean(losses["malignancy"]),
            "loss_total": np.mean(losses["total"]),
            "malignancy_auc": skl_metrics.roc_auc_score(
                np.array(labels["malignancy"]), np.array(predictions["malignancy"])
            ),
            "noduletype_balanced_accuracy": skl_metrics.balanced_accuracy_score(
                labels["noduletype"], [p.argmax() for p in predictions["noduletype"]]
            ),
            "segmentation_dice": 1 - np.mean(losses["segmentation"]),
        }

    def validation(self):
        self.model.eval()
        losses = defaultdict(list)
        predictions = defaultdict(list)
        labels = defaultdict(list)

        with torch.no_grad():
            for batch in tqdm(self.dataloader_valid, desc="Validation"):
                batch_predictions, batch_labels, batch_loss = self.call_model(batch)

                # Store the predictions, labels and losses to later aggregate them.
                for loss, value in batch_loss.items():
                    losses[loss].append(value.item())
                for task, value in batch_predictions.items():
                    predictions[task].extend(value)
                for task, value in batch_labels.items():
                    labels[task].extend(value)

        return {
            "loss_segmentation": np.mean(losses["segmentation"]),
            "loss_noduletype": np.mean(losses["noduletype"]),
            "loss_malignancy": np.mean(losses["malignancy"]),
            "loss_total": np.mean(losses["total"]),
            "malignancy_auc": skl_metrics.roc_auc_score(
                labels["malignancy"], predictions["malignancy"]
            ),
            "noduletype_balanced_accuracy": skl_metrics.balanced_accuracy_score(
                labels["noduletype"],
                [p.argmax() for p in predictions["noduletype"]],
            ),
            "segmentation_dice": 1 - np.mean(losses["segmentation"]),
        }

    def train(self):
        metrics = {"train": [], "valid": []}
        best_metric = 1e9
        best_epoch = 0

        for epoch in range(self.epochs):
            print(f"\n\n===== Epoch {epoch + 1} / {self.epochs} =====\n")

            epoch_train_metrics = self.train_epoch()
            metrics["train"].append(epoch_train_metrics)

            # display_metrics = pandas.DataFrame(epoch_train_metrics).round(3)
            # display_metrics.replace(np.nan, "", inplace=True)
            # print(display_metrics.to_markdown(tablefmt="grid"))
            print(epoch_train_metrics)

            epoch_valid_metrics = self.validation()
            metrics["valid"].append(epoch_valid_metrics)

            # display_metrics = pandas.DataFrame(epoch_valid_metrics).round(3)
            # display_metrics.replace(np.nan, "", inplace=True)
            # print(display_metrics.to_markdown(tablefmt="grid"))
            print(epoch_valid_metrics)

            if epoch_valid_metrics["loss_total"] < best_metric:
                print("\n===== Saving best model! =====\n")
                best_metric = epoch_valid_metrics["loss_total"]
                best_epoch = epoch

                torch.save(self.model.state_dict(), self.save_dir / "best_model.pth")
                np.save(self.save_dir / "best_metrics.npy", epoch_valid_metrics)
            else:
                print(f"Model has not improved since epoch {best_epoch + 1}")

            np.save(self.save_dir / "metrics.npy", metrics)
