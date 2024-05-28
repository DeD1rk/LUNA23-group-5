import json
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.ndimage as ndi
import scipy.ndimage as snd
import SimpleITK as sitk
import torch

from .constants import INPUT_SIZE, NODULETYPE_MAPPING, PATCH_SIZE, PATCH_VOXEL_SPACING
from .dataset import LUNADataset
from .model import Model
from .utils import extract_patch, keep_central_connected_component


def perform_inference_on_one_sample(
    dropout: float = 0.3, checkpoint_name: str = "best-model"
):
    checkpoint_path = Path("/opt/algorithm/checkpoints")
    ct_image_path = list(Path("/input/images/ct/").glob("*"))[0]
    save_path = Path("/output/")

    model = Model(dropout=dropout).cuda()
    model.eval()

    # ⚠️ make sure to adjust these paths
    ckpt = torch.load(checkpoint_path / checkpoint_name / "best_model.pth")
    model.load_state_dict(ckpt)

    segmentation_save_path = save_path / "images" / "lung-nodule-segmentation"
    segmentation_save_path.mkdir(exist_ok=True, parents=True)

    sitk_image = sitk.ReadImage(str(ct_image_path))

    noduleid = ct_image_path.stem
    image = sitk_image
    metad = {
        "origin": np.flip(image.GetOrigin()),
        "spacing": np.flip(image.GetSpacing()),
        "transform": np.array(np.flip(image.GetDirection())).reshape(3, 3),
        "shape": np.flip(image.GetSize()),
    }
    image = sitk.GetArrayFromImage(image)

    image = extract_patch(
        raw_image=image,
        coord=tuple(np.array(INPUT_SIZE) // 2),
        srcWorldMatrix=metad["transform"],
        srcVoxelSpacing=metad["spacing"],
        output_shape=PATCH_SIZE,
        voxel_spacing=PATCH_VOXEL_SPACING,
    )

    image = image.reshape(1, 1, *PATCH_SIZE).astype(np.float32)
    image = LUNADataset.scale_intensity(image)
    image = torch.from_numpy(image).cuda()

    with torch.no_grad():
        outputs = model(image)

    outputs = {
        task: output.detach().cpu().numpy().squeeze()
        for task, output in outputs.items()
    }

    # post-process segmentation

    # resample image to original spacing
    segmentation = ndi.zoom(
        outputs["segmentation"],
        PATCH_VOXEL_SPACING[0] / metad["spacing"],
        order=1,
    )

    # pad image
    diff = metad["shape"] - segmentation.shape
    pad_widths = [
        (np.round(a), np.round(b))
        for a, b in zip(
            diff // 2.0 + 1,
            diff - diff // 2.0 - 1,
        )
    ]
    pad_widths = np.array(pad_widths).astype(int)
    pad_widths = np.clip(pad_widths, 0, pad_widths.max())
    segmentation = np.pad(
        segmentation,
        pad_width=pad_widths,
        mode="constant",
        constant_values=0,
    )

    # crop, if necessary
    if diff.min() < 0:
        shape = np.array(segmentation.shape)
        center = shape // 2

        segmentation = segmentation[
            center[0] - INPUT_SIZE[0] // 2 : center[0] + INPUT_SIZE[0] // 2,
            center[1] - INPUT_SIZE[1] // 2 : center[1] + INPUT_SIZE[1] // 2,
            center[2] - INPUT_SIZE[2] // 2 : center[2] + INPUT_SIZE[2] // 2,
        ]

    # apply threshold
    threshold_value = 0.5
    segmentation = (segmentation > threshold_value).astype(np.uint8)
    segmentation = keep_central_connected_component(segmentation)

    # set metadata
    segmentation = sitk.GetImageFromArray(segmentation)
    segmentation.SetOrigin(np.flip(metad["origin"]))
    segmentation.SetSpacing(np.flip(metad["spacing"]))
    segmentation.SetDirection(np.flip(metad["transform"].reshape(-1)))

    # write as simpleitk image
    sitk.WriteImage(
        segmentation,
        str(segmentation_save_path / f"{noduleid}.mha"),
        True,
    )

    # combine predictions from other task models
    prediction = {
        "noduleid": noduleid,
        "malignancy": outputs["malignancy"],
        "noduletype": outputs["noduletype"].argmax(),
        "ggo_probability": outputs["noduletype"][
            NODULETYPE_MAPPING["GroundGlassOpacity"]
        ],
        "partsolid_probability": outputs["noduletype"][NODULETYPE_MAPPING["SemiSolid"]],
        "solid_probability": outputs["noduletype"][NODULETYPE_MAPPING["Solid"]],
        "calcified_probability": outputs["noduletype"][NODULETYPE_MAPPING["Calcified"]],
    }

    with open("/output/lung-nodule-malignancy-risk.json", "w") as f:
        json.dump(float(prediction["malignancy"]), f)

    with open("/output/lung-nodule-type.json", "w") as f:
        json.dump(int(prediction["noduletype"]), f)


if __name__ == "__main__":
    perform_inference_on_one_sample()
