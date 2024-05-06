import torch
from torch import nn

from .constants import PATCH_SIZE


def conv3x3(n_in, n_out, padding=1):
    return [
        nn.Conv3d(n_in, n_out, kernel_size=3, padding=padding, bias=False),
        nn.BatchNorm3d(n_out, affine=True, track_running_stats=False),
        nn.ReLU(inplace=True),
    ]


class Flatten(nn.Module):
    def forward(self, y):
        return y.view(y.size(0), -1)


class Model(nn.Module):
    """A multi-task model for segmentation and classification of lung nodules."""

    def __init__(self):
        super().__init__()

        # The baseline CNN3D for malignancy.
        self.layers = nn.Sequential(
            *conv3x3(1, 32, padding=0),
            nn.MaxPool3d(kernel_size=2),
            *conv3x3(32, 64, padding=0),
            nn.MaxPool3d(kernel_size=2),
            *conv3x3(64, 64, padding=0),
            nn.MaxPool3d(kernel_size=2),
            *conv3x3(64, 128, padding=0),
            nn.MaxPool3d(kernel_size=2),
            Flatten(),
            nn.Linear(1024, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        outputs = self.layers(x)
        # Return a dictionary with the outputs for each task.
        return {
            # Placeholders with correct shapes.
            "segmentation": torch.zeros((x.shape[0],) + PATCH_SIZE).to(
                outputs.get_device()
            ),
            "malignancy": outputs.squeeze(),
            "noduletype": torch.zeros(x.shape[0], 4).to(outputs.get_device()),
        }
