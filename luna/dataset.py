from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK
from tqdm import tqdm


class NoduleType(Enum):
    GroundGlassOpacity = 1
    Calcified = 2
    SemiSolid = 3
    Solid = 4

NODULE_TYPE_MAPPING = {
    "GroundGlassOpacity": NoduleType.GroundGlassOpacity,
    "Calcified": NoduleType.Calcified,
    "SemiSolid": NoduleType.SemiSolid,
    "Solid": NoduleType.Solid,
}

def load_dataset(always_parse=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse or open the training dataset as several numpy arrays.

    By default, this looks for a generated .npz file, and opens it if it exists.
    If it does not exist, or `always_parse=True`, the full dataset is parsed
    and saved to disk for later reuse.
    """
    generated_dir = Path(f"dataset/generated")
    generated_dataset = generated_dir / "train.npz"
    if always_parse or not generated_dataset.exists():
        # Parse the dataset from the original files.
        df = pd.read_csv(Path("dataset/luna23-ismi-train-set.csv"))
        size = len(df)
        images = np.zeros((size, 64, 128, 128), dtype=np.float32)
        labels_segmentation = np.zeros((size, 64, 128, 128), dtype=np.uint8)
        labels_type = np.zeros(size, dtype=np.uint8)
        labels_malignancy = np.zeros(size, dtype=np.uint8)
        for index, row in tqdm(
            df.iterrows(), desc="Loading training files", total=size
        ):
            images[index] = SimpleITK.GetArrayFromImage(
                SimpleITK.ReadImage(Path("dataset/train_set/images") / f"{row['noduleid']}.mha")
            )
            labels_segmentation[index] =  SimpleITK.GetArrayFromImage(
                SimpleITK.ReadImage(Path("dataset/train_set/labels") / f"{row['noduleid']}.mha")
            )
            labels_type[index], labels_malignancy[index] = (
                NODULE_TYPE_MAPPING[row["noduletype"]].value,
                row["malignancy"],
            )

        # Save the parsed data.
        generated_dir.mkdir(exist_ok=True, parents=False)
        np.savez(
            generated_dataset,
            images=images,
            labels_segmentation=labels_segmentation,
            labels_malignancy=labels_malignancy,
            labels_type=labels_type,
        )
    else:
        print("Reading parsed data from saved arrays.")
        # Load parsed data from saved numpy arrays.
        data = np.load(generated_dataset)
        images, labels_segmentation, labels_malignancy, labels_type = (
            data["images"],
            data["labels_segmentation"],
            data["labels_malignancy"],
            data["labels_type"],
        )
        data.close()

    return images, labels_segmentation, labels_malignancy, labels_type

if __name__ == "__main__":
    load_dataset()