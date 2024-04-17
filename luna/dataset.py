from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK
from tqdm import tqdm


class NoduleType(Enum):
    NonSolid = 1
    PartSolid = 2
    Solid = 3


# We resample to the min spacing accross all of the training images.
SPACING = (0.49609375, 0.49609375, 0.5)

# TODO: check this mapping. What is Calcified and what is SolidSpiculated (does it even occur)?
NODULE_TYPE_MAPPING = {
    "GroundGlassOpacity": NoduleType.NonSolid,
    "Calcified": NoduleType.NonSolid,
    "SemiSolid": NoduleType.PartSolid,
    "SolidSpiculated": NoduleType.Solid,  # This does not appear in the training data.
    "Solid": NoduleType.Solid,
}


def load_image(filename: Path, new_spacing=SPACING) -> np.ndarray:
    """Parse and resample an image.

    Loads a .mha file, resamples it, and returns a (64,128,128) numpy array of float32.
    The output is a resampled and cropped image, centered on the center of the original.
    """
    image = SimpleITK.ReadImage(str(filename), SimpleITK.sitkFloat32)

    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_size = [
        int(np.ceil(osz * ospc / nspc))
        for osz, ospc, nspc in zip(old_size, old_spacing, new_spacing)
    ]

    # Resample to a larger array covering the whole input area.
    resampled_image = SimpleITK.Resample(
        image,
        new_size,
        SimpleITK.Transform(),
        SimpleITK.sitkLinear,
        image.GetOrigin(),
        new_spacing,
        image.GetDirection(),
        0,
        image.GetPixelID(),
    )
    resampled_array = SimpleITK.GetArrayFromImage(resampled_image)

    # Crop the array to (64,128,128) around the center.
    centers = [int(round(x / 2)) for x in new_size]
    cropped = np.copy(
        resampled_array[
            # Reorder dimensions from SimpleITK's (x,y,z) to numpy's (z,y,x)
            centers[2] - 32 : centers[2] + 32,
            centers[1] - 64 : centers[1] + 64,
            centers[0] - 64 : centers[0] + 64,
        ]
    )

    return cropped


def load_dataset(always_parse=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse or open the training dataset as several numpy arrays.

    By default, this looks for a generated .npz file, and opens it if it exists.
    If it does not, or `always_parse=True`, the full dataset is parsed, resampled,
    and saved to disk for later reuse.
    """
    generated_path = Path("dataset/train.npz")
    if always_parse or not generated_path.exists():
        # Parse the dataset from the original files.
        df = pd.read_csv(Path("dataset/luna23-ismi-train-set.csv"))
        size = len(df)
        images = np.zeros((size, 64, 128, 128), dtype=np.float32)
        labels_type = np.zeros(size, dtype=np.uint8)
        labels_malignancy = np.zeros(size, dtype=np.uint8)
        for index, row in tqdm(
            df.iterrows(), desc="Loading training files", total=size
        ):
            filename = (
                Path("dataset") / "train_set" / "images" / f"{row['noduleid']}.mha"
            )
            images[index] = load_image(filename)
            labels_type[index], labels_malignancy[index] = (
                NODULE_TYPE_MAPPING[row["noduletype"]].value,
                row["malignancy"],
            )

        # Save the parsed data.
        np.savez(
            generated_path,
            images=images,
            labels_malignancy=labels_malignancy,
            labels_type=labels_type,
        )
    else:
        # Load parsed data from saved numpy arrays.
        data = np.load(generated_path)
        images, labels_malignancy, labels_type = (
            data["images"],
            data["labels_malignancy"],
            data["labels_type"],
        )
        data.close()

    return images, labels_malignancy, labels_type
