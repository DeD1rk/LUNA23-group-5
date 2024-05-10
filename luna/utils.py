from pathlib import Path

import numpy as np
import numpy.linalg as npl
import pandas as pd
import scipy.ndimage as ndi
import SimpleITK as sitk
import torch
from sklearn.model_selection import StratifiedKFold

from .constants import PATCH_SIZE, PATCH_VOXEL_SPACING


def make_development_splits(data_dir: Path, n_folds: int = 5):
    """Split the training set into folds at a patient-level."""
    np.random.seed(2023)

    train_set = pd.read_csv(data_dir / "luna23-ismi-train-set.csv")

    save_path = data_dir / "folds"
    save_path.mkdir(exist_ok=True, parents=True)

    pids = train_set.patientid.unique()
    labs = [train_set[train_set.patientid == pid].malignancy.values[0] for pid in pids]
    labs = np.array(labs)

    assert len(pids) == len(labs)

    skf = StratifiedKFold(n_splits=n_folds)
    skf.get_n_splits(pids, labs)

    folds_missing = False

    for fold in range(n_folds):
        train_pd = save_path / f"train{fold}.csv"
        valid_pd = save_path / f"valid{fold}.csv"

        if not train_pd.is_file():
            folds_missing = True

        if not valid_pd.is_file():
            folds_missing = True

    if folds_missing:
        print(f"Making {n_folds} folds from the train set")

        for fold, (train_index, test_index) in enumerate(skf.split(pids, labs)):
            train_pids, valid_pids = pids[train_index], pids[test_index]

            train_pd = train_set[train_set.patientid.isin(train_pids)]
            valid_pd = train_set[train_set.patientid.isin(valid_pids)]

            train_pd = train_pd.reset_index(drop=True)
            valid_pd = valid_pd.reset_index(drop=True)

            train_pd.to_csv(save_path / f"train{fold}.csv", index=False)
            valid_pd.to_csv(save_path / f"valid{fold}.csv", index=False)


def _calculateAllPermutations(itemList):
    if len(itemList) == 1:
        return [[i] for i in itemList[0]]
    else:
        sub_permutations = _calculateAllPermutations(itemList[1:])
        return [[i] + p for i in itemList[0] for p in sub_permutations]


def sample_random_coordinate_on_sphere(radius):
    # Generate three random numbers x,y,z using Gaussian distribution
    random_nums = np.random.normal(size=(3,))

    # You should handle what happens if x=y=z=0.
    if np.all(random_nums == 0):
        return np.zeros((3,))

    # Normalise numbers and multiply number by radius of sphere
    return random_nums / np.sqrt(np.sum(random_nums * random_nums)) * radius


def volumeTransform(
    image,
    voxel_spacing,
    transform_matrix,
    center=None,
    output_shape=None,
    output_voxel_spacing=None,
    **argv,
):
    """
    Parameters
    ----------
      image : a numpy.ndarray
          The image that should be transformed

      voxel_spacing : a vector
          This vector describes the voxel spacing between individual pixels. Can
          be filled with (1,) * image.ndim if unknown.

      transform_matrix : a Nd x Nd matrix where Nd is the number of image dimensions
          This matrix governs how the output image will be oriented. The x-axis will be
          oriented along the last row vector of the transform_matrix, the y-Axis along
          the second-to-last row vector etc. (Note that numpy uses a matrix ordering
          of axes to index image axes). The matrix must be square and of the same
          order as the dimensions of the input image.

          Typically, this matrix is the transposed mapping matrix that maps coordinates
          from the projected image to the original coordinate space.

      center : vector (default: None)
          The center point around which the transform_matrix pivots to extract the
          projected image. If None, this defaults to the center point of the
          input image.

      output_shape : a list of integers (default None)
          The shape of the image projection. This can be used to limit the number
          of pixels that are extracted from the orignal image. Note that the number
          of dimensions must be equal to the number of dimensions of the
          input image. If None, this defaults to dimenions needed to enclose the
          whole inpput image given the transform_matrix, center, voxelSPacings,
          and the output_shape.

      output_voxel_spacing : a vector (default: None)
          The interleave at which points should be extracted from the original image.
          None, lets the function default to a (1,) * output_shape.ndim value.

      **argv : extra arguments
          These extra arguments are passed directly to scipy.ndimage.affine_transform
          to allow to modify its behavior. See that function for an overview of optional
          paramters (other than offset and output_shape which are used by this function
          already).
    """
    if "offset" in argv:
        raise ValueError(
            "Cannot supply 'offset' to scipy.ndimage.affine_transform - already used by this function"
        )
    if "output_shape" in argv:
        raise ValueError(
            "Cannot supply 'output_shape' to scipy.ndimage.affine_transform - already used by this function"
        )

    if image.ndim != len(voxel_spacing):
        raise ValueError("Voxel spacing must have the same dimensions")

    if center is None:
        voxelCenter = (np.array(image.shape) - 1) / 2.0
    else:
        if len(center) != image.ndim:
            raise ValueError(
                "center point has not the same dimensionality as the image"
            )

        # Transform center to voxel coordinates
        voxelCenter = np.asarray(center) / voxel_spacing

    transform_matrix = np.asarray(transform_matrix)
    if output_voxel_spacing is None:
        if output_shape is None:
            output_voxel_spacing = np.ones(transform_matrix.shape[0])
        else:
            output_voxel_spacing = np.ones(len(output_shape))
    else:
        output_voxel_spacing = np.array(output_voxel_spacing)

    if transform_matrix.shape[1] != image.ndim:
        raise ValueError(
            "transform_matrix does not have the correct number of columns (does not match image dimensionality)"
        )
    if transform_matrix.shape[0] != image.ndim:
        raise ValueError(
            "Only allowing square transform matrices here, even though this is unneccessary. However, one will need an algorithm here to create full rank-square matrices. 'QR decomposition with Column Pivoting' would probably be a solution, but the author currently does not know what exactly this is, nor how to do this..."
        )
    #  print (transform_matrix, transform_matrix, np.zeros((transform_matrix.shape[1], image.ndim - transform_matrix.shape[0])))
    #  transform_matrix = np.hstack((transform_matrix, np.zeros((transform_matrix.shape[1], image.ndim - transform_matrix.shape[0]))))

    # Normalize the transform matrix
    transform_matrix = np.array(transform_matrix)
    transform_matrix = (
        transform_matrix.T
        / np.sqrt(np.sum(transform_matrix * transform_matrix, axis=1))
    ).T
    transform_matrix = np.linalg.inv(
        transform_matrix.T
    )  # Important normalization for shearing matrices!!

    # The forwardMatrix transforms coordinates from input image space into result image space
    forward_matrix = np.dot(
        np.dot(np.diag(1.0 / output_voxel_spacing), transform_matrix),
        np.diag(voxel_spacing),
    )

    if output_shape is None:
        # No output dimensions are specified
        # Therefore we calculate the region that will span the whole image
        # considering the transform matrix and voxel spacing.
        image_axes = [[0 - o, x - 1 - o] for o, x in zip(voxelCenter, image.shape)]
        image_corners = _calculateAllPermutations(image_axes)

        transformed_image_corners = map(
            lambda x: np.dot(forward_matrix, x), image_corners
        )
        output_shape = [
            1 + int(np.ceil(2 * max(abs(x_max), abs(x_min))))
            for x_min, x_max in zip(
                np.amin(transformed_image_corners, axis=0),
                np.amax(transformed_image_corners, axis=0),
            )
        ]
    else:
        # Check output_shape
        if len(output_shape) != transform_matrix.shape[1]:
            raise ValueError(
                "output dimensions must match dimensionality of the transform matrix"
            )
    output_shape = np.array(output_shape)

    # Calculate the backwards matrix which will be used for the slice extraction
    backwards_matrix = npl.inv(forward_matrix)
    target_image_offset = voxelCenter - backwards_matrix.dot((output_shape - 1) / 2.0)

    return ndi.affine_transform(
        image,
        backwards_matrix,
        offset=target_image_offset,
        output_shape=output_shape,
        **argv,
    )


def rotateMatrixX(cosAngle, sinAngle):
    return np.asarray([[1, 0, 0], [0, cosAngle, -sinAngle], [0, sinAngle, cosAngle]])


def rotateMatrixY(cosAngle, sinAngle):
    return np.asarray([[cosAngle, 0, sinAngle], [0, 1, 0], [-sinAngle, 0, cosAngle]])


def rotateMatrixZ(cosAngle, sinAngle):
    return np.asarray([[cosAngle, -sinAngle, 0], [sinAngle, cosAngle, 0], [0, 0, 1]])


def worker_init_fn(worker_id):
    """Seed numpy random state with unique seed for each worker."""
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32)
    np.random.seed(seed=seed)


def extract_patch(
    raw_image: np.ndarray,
    coord,
    srcVoxelOrigin,
    srcWorldMatrix,
    srcVoxelSpacing,
    mask: np.ndarray | None = None,
    output_shape=PATCH_SIZE,
    voxel_spacing=PATCH_VOXEL_SPACING,
    rotations=None,
    translations=None,
    mirrorings=None,
    coord_space_world=False,
    offset=np.array([0, 0, 0]),
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    # Start with the identity linear transformation matrix.
    transform_matrix = np.eye(3)

    # Apply random rotation matrix.
    if rotations is not None:
        (zmin, zmax), (ymin, ymax), (xmin, xmax) = rotations

        # Determine rotation angles.
        angleX = np.multiply(np.pi / 180.0, np.random.randint(xmin, xmax, 1))[0]
        angleY = np.multiply(np.pi / 180.0, np.random.randint(ymin, ymax, 1))[0]
        angleZ = np.multiply(np.pi / 180.0, np.random.randint(zmin, zmax, 1))[0]

        # Multiply rotation matrices for each axis.
        rotation_matrix = np.eye(3)
        rotation_matrix = np.dot(
            rotation_matrix, rotateMatrixX(np.cos(angleX), np.sin(angleX))
        )
        rotation_matrix = np.dot(
            rotation_matrix, rotateMatrixY(np.cos(angleY), np.sin(angleY))
        )
        rotation_matrix = np.dot(
            rotation_matrix, rotateMatrixZ(np.cos(angleZ), np.sin(angleZ))
        )

        # Apply rotation matrix.
        transform_matrix = np.dot(transform_matrix, rotation_matrix)

    # Apply random mirroring matrix.
    if mirrorings is not None:
        mirroring_matrix = np.eye(3)
        for axis in range(3):
            if mirrorings[axis] and np.random.random() > 0.5:
                mirroring_matrix[axis, axis] = -1

        transform_matrix = np.dot(transform_matrix, mirroring_matrix)

    # compute random translation
    if translations is not None:
        # add random translation
        radius = np.random.random_sample() * translations
        offset = sample_random_coordinate_on_sphere(radius=radius)
        offset = offset * (1.0 / srcVoxelSpacing)

    # apply random translation
    coord = np.array(coord) + offset

    thisTransformMatrix = transform_matrix
    # Normalize transform matrix
    thisTransformMatrix = (
        thisTransformMatrix.T
        / np.sqrt(np.sum(thisTransformMatrix * thisTransformMatrix, axis=1))
    ).T

    invSrcMatrix = np.linalg.inv(srcWorldMatrix)

    # world coord sampling
    if coord_space_world:
        overrideCoord = invSrcMatrix.dot(coord - srcVoxelOrigin)
    else:
        # image coord sampling
        overrideCoord = coord * srcVoxelSpacing
    overrideMatrix = (invSrcMatrix.dot(thisTransformMatrix.T) * srcVoxelSpacing).T

    patch = volumeTransform(
        raw_image,
        srcVoxelSpacing,
        overrideMatrix,
        center=overrideCoord,
        output_shape=np.array(output_shape),
        output_voxel_spacing=np.array(voxel_spacing),
        order=1,
        prefilter=False,
    )
    patch = np.expand_dims(patch, axis=0)

    if mask is not None:
        mask = volumeTransform(
            mask,
            srcVoxelSpacing,
            overrideMatrix,
            center=overrideCoord,
            output_shape=np.array(output_shape),
            output_voxel_spacing=np.array(voxel_spacing),
            order=0,
            prefilter=False,
        )
        mask = np.expand_dims(mask, axis=0)
        return patch, mask

    else:
        return patch


def keep_central_connected_component(
    prediction: sitk.Image,
) -> sitk.Image:
    """Function to post-process the prediction to keep only the central connected component in a patch

    Args:
        prediction (sitk.Image): prediction file (should be binary)
        patch_size (np.array, optional): patch size (x, y, z) to ensure the center is computed appropriately.

    Returns:
        sitk.Image: post-processed binary file with only the central connected component
    """

    origin = prediction.GetOrigin()
    spacing = prediction.GetSpacing()
    direction = prediction.GetDirection()

    prediction = sitk.GetArrayFromImage(prediction)

    c, n = ndi.label(prediction)
    centroids = np.array(
        [np.array(np.where(c == i)).mean(axis=1) for i in range(1, n + 1)]
    ).astype(int)

    patch_size = np.array(list(reversed(PATCH_SIZE)))

    if len(centroids) > 0:
        dists = np.sqrt(((centroids - patch_size // 2) ** 2).sum(axis=1))
        keep_idx = np.argmin(dists)
        output = np.zeros(c.shape)
        output[c == (keep_idx + 1)] = 1
        prediction = output.astype(np.uint8)

    prediction = sitk.GetImageFromArray(prediction)
    prediction.SetSpacing(spacing)
    prediction.SetOrigin(origin)
    prediction.SetDirection(direction)
    return prediction
