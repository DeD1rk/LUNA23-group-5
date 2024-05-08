from pathlib import Path

import numpy as np
import pandas
import scipy.ndimage as ndi
import SimpleITK as sitk
import torch
from tqdm import tqdm

from .constants import INPUT_SIZE, PATCH_SIZE, PATCH_VOXEL_SPACING
from .dataset import LUNADataset
from .model import Model
from .utils import extract_patch, keep_central_connected_component


def perform_inference_on_test_set(data_dir: Path, result_dir: Path):
    model = Model().cuda()
    model.eval()

    checkpoint = torch.load(result_dir / "best_model.pth")
    model.load_state_dict(checkpoint)

    test_set_path = Path(data_dir / "test_set" / "images")
    save_path = result_dir / "test_set_predictions"

    segmentation_save_path = save_path / "segmentations"
    segmentation_save_path.mkdir(exist_ok=True, parents=True)

    predictions = []

    for image_path in tqdm(list(test_set_path.glob("*.mha"))):
        # load and pre-process input image
        sitk_image = sitk.ReadImage(image_path)

        noduleid = image_path.stem
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
            srcVoxelOrigin=(0, 0, 0),
            srcWorldMatrix=metad["transform"],
            srcVoxelSpacing=metad["spacing"],
            output_shape=PATCH_SIZE,
            voxel_spacing=PATCH_VOXEL_SPACING,
            coord_space_world=False,
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
        segmentation = (segmentation > 0.5).astype(np.uint8)

        # set metadata
        segmentation = sitk.GetImageFromArray(segmentation)
        segmentation.SetOrigin(np.flip(metad["origin"]))
        segmentation.SetSpacing(np.flip(metad["spacing"]))
        segmentation.SetDirection(np.flip(metad["transform"].reshape(-1)))

        # keep central connected component
        segmentation = keep_central_connected_component(segmentation)

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
            "ggo_probability": outputs["noduletype"][0],
            "partsolid_probability": outputs["noduletype"][1],
            "solid_probability": outputs["noduletype"][2],
            "calcified_probability": outputs["noduletype"][3],
        }

        predictions.append(pandas.Series(prediction))

    predictions = pandas.DataFrame(predictions)
    predictions.to_csv(save_path / "predictions.csv", index=False)
