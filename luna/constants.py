NODULETYPE_MAPPING = {
    "GroundGlassOpacity": 0,
    "SemiSolid": 1,
    "Solid": 2,
    "Calcified": 3,
}

HU_MIN, HU_MAX = -1000, 400

# Size of input images.
INPUT_SIZE = (64, 128, 128)

# Size of patches after preprocessing.
PATCH_SIZE = (64, 64, 64)

# Voxel spacing in the preprocessed patches.
PATCH_VOXEL_SPACING = (50 / 64, 50 / 64, 50 / 64)
