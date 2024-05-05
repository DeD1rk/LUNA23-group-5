import torch

from .constants import PATCH_SIZE


class Model(torch.nn.Module):
    """A multi-task model for segmentation and classification of lung nodules."""

    def __init__(self):
        super().__init__()
        # Create placeholder model structure with just a single linear layer.
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(64**3, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        outputs = self.layers(x.view(x.size(0), -1))
        # Return a dictionary with the outputs for each task.
        return {
            # Placeholders with correct shapes.
            "segmentation": torch.zeros((x.shape[0],) + PATCH_SIZE),
            "malignancy": outputs.squeeze(),
            "noduletype": torch.zeros(x.shape[0], 4),
        }
