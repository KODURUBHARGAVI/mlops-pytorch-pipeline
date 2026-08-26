"""Model definitions for the CIFAR-10 image classifier.

Two architectures
simple_cnn- a small from-scratch CNN, useful for fast CPU smoke tests.
resnet18- torchvision ResNet-18 adapted for 32x32 CIFAR images.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

__all__ = ["SimpleCNN", "get_model", "SUPPORTED_ARCHITECTURES"]

SUPPORTED_ARCHITECTURES = ("simple_cnn", "resnet18")


class SimpleCNN(nn.Module):
    """A compact 3-block CNN. Global pooling makes it input-size agnostic."""

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
    """ResNet-18 re-wired for small CIFAR-sized inputs.

    The stock torchvision stem (7x7 stride-2 conv + maxpool) throws away far
    too much spatial information on a 32x32 image, so it is replaced with a
    3x3 stride-1 conv and the maxpool is removed.
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
    """Return an un-trained model for ``architecture``.

    Args:
        architecture: ``simple_cnn`` or ``resnet18``.
        num_classes: Width of the classification head.
        pretrained: Load ImageNet weights (resnet18 only). Needs network access
            at build time, so it stays off by default.
        in_channels: 3 for RGB (CIFAR-10), 1 for grayscale (Fashion-MNIST).

    Raises:
        ValueError: if the architecture name is not supported.
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
