import torch
from torch import nn


def conv3x3(n_in, n_out, padding=1):
    return [
        nn.Conv3d(n_in, n_out, kernel_size=3, padding=padding, bias=False),
        nn.BatchNorm3d(n_out, affine=True),
        nn.ReLU(inplace=True),
    ]


class ContractionBlock(nn.Module):
    def __init__(self, n_input_channels, n_filters, dropout=None, pooling=True):
        super().__init__()

        layers = []
        if pooling:
            layers.append(nn.MaxPool3d(kernel_size=2))
        layers += conv3x3(n_input_channels, n_filters)
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers += conv3x3(n_filters, n_filters)
        self.pool_conv = nn.Sequential(*layers)

    def forward(self, incoming):
        return self.pool_conv(incoming)


class ExpansionBlock(nn.Module):
    def __init__(self, n_input_channels, n_filters, dropout=None):
        super(ExpansionBlock, self).__init__()

        self.upconv = nn.Sequential(
            nn.ConvTranspose3d(
                n_input_channels,
                n_filters,
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm3d(n_filters, affine=True, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

        layers = conv3x3(n_filters * 2, n_filters)
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        layers += conv3x3(n_filters, n_filters)
        self.conv = nn.Sequential(*layers)

    def forward(self, incoming, skip_connection):
        y = self.upconv(incoming)
        y = torch.cat([y, skip_connection], dim=1)
        return self.conv(y)


class Flatten(nn.Module):
    def forward(self, y):
        return y.view(y.size(0), -1)


class Model(nn.Module):
    """A multi-task model for segmentation and classification of lung nodules."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()

        self.encoder = nn.ModuleList(
            [
                # Input shape: (64x64x64 @ 1)
                ContractionBlock(1, 16, dropout=None, pooling=False),
                ContractionBlock(16, 32, dropout=dropout, pooling=True),
                ContractionBlock(32, 64, dropout=dropout, pooling=True),
                ContractionBlock(64, 64, dropout=dropout, pooling=True),
                ContractionBlock(64, 64, dropout=dropout, pooling=True),
                # Output shape: (4x4x4 @ 64)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                # Input shape: (4x4x4 @ 64)
                ExpansionBlock(64, 64, dropout=dropout),
                ExpansionBlock(64, 64, dropout=dropout),
                ExpansionBlock(64, 64, dropout=dropout),
                ExpansionBlock(64, 32, dropout=dropout),
                ExpansionBlock(32, 16, dropout=dropout),
                # Output shape: (64x64x64 @ 16)
            ]
        )

        self.segmentation = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.shared_classification = nn.Sequential(
            nn.Conv3d(64, 32, 3),
            nn.ReLU(),
            Flatten(),
            nn.Linear(4 * 4 * 4 * 16, 128),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
        )

        self.malignancy = nn.Sequential(
            nn.Linear(128, out_features=1),
            nn.Sigmoid(),
        )

        self.noduletype = nn.Sequential(
            nn.Linear(128, out_features=4),
            nn.Softmax(dim=1),
        )

    def forward(self, image):
        y = image

        contraction_states = []
        for contraction_block in self.encoder:
            y = contraction_block(y)
            contraction_states.append(y)

        for expansion_block, skip_state in zip(
            self.decoder, reversed(contraction_states[:-1])
        ):
            y = expansion_block(y, skip_state)

        shared_classification_state = self.shared_classification(contraction_states[-1])

        # Return a dictionary with the outputs for each task.
        return {
            "segmentation": self.segmentation(y),
            "malignancy": self.malignancy(shared_classification_state).squeeze(),
            "noduletype": self.noduletype(shared_classification_state),
        }
