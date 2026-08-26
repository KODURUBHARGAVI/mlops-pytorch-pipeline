"""The model designs used in this project.

simple_cnn is a small CNN written from scratch. resnet18 is the torchvision
model, adjusted to work with small images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

__all__ = ["SimpleCNN", "get_model", "SUPPORTED_ARCHITECTURES"]

SUPPORTED_ARCHITECTURES = ("simple_cnn", "resnet18")


class SimpleCNN(nn.Module):
    """A small CNN with 3 blocks. Pooling at the end makes the image size flexible."""

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(in_channels, 32),
            self._block(32, 64),
            self._block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _build_resnet18(
    num_classes: int, pretrained: bool = False, in_channels: int = 3
) -> nn.Module:
    """ResNet-18 adjusted for small images.

    The first layer in torchvision uses a large filter and a pooling step made
    for big photographs, which removes too much detail from a 28 by 28 image.
    Both are replaced here.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
    in_channels: int = 3,
) -> nn.Module:
    """Returns an untrained model.

    architecture is simple_cnn or resnet18, and in_channels is 1 for grey
    images or 3 for colour ones. An unknown architecture raises ValueError.
    """
    name = architecture.lower().strip()
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    if name == "resnet18":
        return _build_resnet18(
            num_classes=num_classes, pretrained=pretrained, in_channels=in_channels
        )
    raise ValueError(
        f"Unsupported architecture {architecture!r}. "
        f"Expected one of: {', '.join(SUPPORTED_ARCHITECTURES)}"
    )
