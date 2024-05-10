from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import (
    HU_MAX,
    HU_MIN,
    INPUT_SIZE,
    NODULETYPE_MAPPING,
    PATCH_SIZE,
    PATCH_VOXEL_SPACING,
)
from .utils import extract_patch, make_development_splits


def get_balancing_weights(labels) -> np.ndarray:
    """Return sampling weights to mitigate class imbalance."""
    _, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / counts
    return weights[labels]


class LUNADataset(Dataset):
    """A dataset returning center-cropped patches of nodules.

    This dataset class loads all of the images into memory to avoid slow disk access.
    """

    def __init__(
        self,
        data_dir: Path,
        fold: int = 0,
        validation: bool = False,
        max_rotation_degrees: float = 0,
        enable_translations: bool = False,
        enable_mirroring: tuple[bool, bool, bool] = (False, False, False),
    ):
        """Create an instance of the dataset.

        The path to the root of the dataset should be provided. Based on that,
        the training of validation set for the provided `fold` will be loaded.
        If no folds have been made yet, this wil also create a 5-fold split.

        The keyword arguments `max_rotation_degrees`, `enable_translations` and
        `enable_mirroring` can be used to control data augmentation in the training
        dataset. If `max_rotation_degrees` is set to a value greater than 0, random
        rotations will be applied in each axis. If `enable_translations` is True,
        the nodule will be randomly translated within a sphere with a radius equal
        to the nodule's diameter. `enable_mirroring` is a tuple of three booleans, each
        for enabling mirroring along the z (head-toe), y (front-back) and x (left-right)
        axis respectively.
        """
        self.data_dir = data_dir
        self.fold = fold
        self.validation = validation

        self.enable_translations = enable_translations
        self.enable_mirroring = enable_mirroring  # Mirroring flag for each axis.
        self.rotations = (  # Rotation range for each axis.
            [(-max_rotation_degrees, max_rotation_degrees)] * 3
            if max_rotation_degrees > 0
            else None
        )

        df_path = (
            self.data_dir / "folds" / f"{'valid' if validation else 'train'}{fold}.csv"
        )
        if not df_path.exists():
            make_development_splits(data_dir)

        self.dataframe = pd.read_csv(df_path)
        self._load_images()

    def _load_images(self):
        size = len(self.dataframe)
        self._raw_images = np.zeros((size,) + INPUT_SIZE, dtype=np.float32)
        self._raw_segmentation_labels = np.zeros((size,) + INPUT_SIZE, dtype=np.uint8)
        self._noduletype_labels = np.zeros(size, dtype=np.uint8)
        self._malignancy_labels = np.zeros(size, dtype=np.uint8)
        self._metadata = []

        for index, row in tqdm(
            self.dataframe.iterrows(),
            desc=f"Loading {'validation' if self.validation else 'training'} files",
            total=size,
        ):
            image = sitk.ReadImage(
                self.data_dir / "train_set" / "images" / f"{row['noduleid']}.mha"
            )
            segmentation = sitk.ReadImage(
                self.data_dir / "train_set" / "labels" / f"{row['noduleid']}.mha"
            )
            self._raw_images[index] = sitk.GetArrayFromImage(image)
            self._raw_segmentation_labels[index] = sitk.GetArrayFromImage(segmentation)

            self._noduletype_labels[index] = NODULETYPE_MAPPING[row["noduletype"]]
            self._malignancy_labels[index] = row["malignancy"]
            self._metadata.append(
                {
                    "origin": np.flip(image.GetOrigin()),
                    "spacing": np.flip(image.GetSpacing()),
                    "transform": np.array(np.flip(image.GetDirection())).reshape(3, 3),
                    "shape": np.flip(image.GetSize()),
                }
            )

    @classmethod
    def scale_intensity(cls, image):
        return np.clip((image - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)

    def _extract_patch(self, index: int) -> tuple[np.ndarray, np.ndarray, dict]:
        image = self._raw_images[index]
        segmentation_label = self._raw_segmentation_labels[index]
        metadata = self._metadata[index]
        dataframe_row = self.dataframe.iloc[index]

        translations = None
        if self.enable_translations:
            # A random translation within a sphere will be applied.
            # Pick the radius based on the (known) radius of the nodule.
            radius = dataframe_row.diameter_mm / 2
            translations = radius if radius > 0 else None

        patch, mask = extract_patch(
            raw_image=image,
            coord=tuple(np.array(INPUT_SIZE) // 2),
            srcVoxelOrigin=(0, 0, 0),
            srcWorldMatrix=metadata["transform"],
            srcVoxelSpacing=metadata["spacing"],
            mask=segmentation_label,
            output_shape=PATCH_SIZE,
            voxel_spacing=PATCH_VOXEL_SPACING,
            rotations=self.rotations,
            translations=translations,
            mirrorings=self.enable_mirroring,
            coord_space_world=False,
        )

        return self.scale_intensity(patch), mask, metadata

    def __getitem__(self, index: int) -> dict:
        dataframe_row = self.dataframe.iloc[index]
        patch, mask, metadata = self._extract_patch(index)

        sample = {
            "image": torch.from_numpy(patch),
            "segmentation_label": torch.from_numpy(mask),
            "malignancy_label": self._malignancy_labels[index],
            "noduletype_label": self._noduletype_labels[index],
            "origin": torch.from_numpy(metadata["origin"].copy()),
            "spacing": torch.from_numpy(metadata["spacing"].copy()),
            "transform": torch.from_numpy(metadata["transform"].copy()),
            "shape": torch.from_numpy(metadata["shape"].copy()),
            "noduleid": dataframe_row.noduleid,
        }

        return sample

    def __len__(self):
        return len(self.dataframe)
