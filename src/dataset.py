"""Loads the Fashion-MNIST images and prepares them for the model.

The training script and the API both use the values here, so an image is
prepared the same way during training and during prediction.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = [
    "DATASET_NAME",
    "CLASSES",
    "NUM_CLASSES",
    "IMAGE_SIZE",
    "IN_CHANNELS",
    "IMAGE_MODE",
    "ConvertImageMode",
    "get_transforms",
    "get_inference_transform",
    "get_dataloaders",
]

DATASET_NAME = "fashionmnist"

IMAGE_SIZE = 28
IN_CHANNELS = 1
IMAGE_MODE = "L"        # the PIL mode for a single channel image
MEAN = (0.2860,)        # average pixel value across the training images
STD = (0.3530,)         # spread of pixel values across the training images
CROP_PADDING = 2

CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)
NUM_CLASSES = len(CLASSES)


class ConvertImageMode:
    """Converts an uploaded image to the colour mode the model expects.

    Uploads can be colour, grey or palette based, and the wrong number of
    channels would fail at the normalisation step.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(self, image):
        return image if image.mode == self.mode else image.convert(self.mode)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mode={self.mode!r})"


def get_transforms(train: bool = True) -> transforms.Compose:
    """Prepares an image for the model. Extra variation is added for training."""
    steps: list = []
    if train:
        # Small random changes so the model does not memorise exact pixels.
        steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.RandomCrop(IMAGE_SIZE, padding=CROP_PADDING))
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def get_inference_transform() -> transforms.Compose:
    """Prepares one uploaded image, which can be any size or colour mode."""
    return transforms.Compose(
        [
            ConvertImageMode(IMAGE_MODE),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def _maybe_subset(dataset, fraction: float, seed: int):
    """Keeps the given fraction of a dataset. A value of 1.0 keeps everything."""
    if fraction >= 1.0:
        return dataset
    if not 0 < fraction < 1:
        raise ValueError(f"subset_fraction must be between 0 and 1, got {fraction}")
    keep = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:keep].tolist()
    return Subset(dataset, indices)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int | None = None,
    download: bool = True,
    subset_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Builds the training and validation loaders.

    data_dir is a mounted folder inside a container, so the images are
    downloaded only once. subset_fraction below 1.0 gives a quicker run.
    """
    if num_workers is None:
        num_workers = int(os.getenv("DATALOADER_WORKERS", "2"))

    os.makedirs(data_dir, exist_ok=True)

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=download,
        transform=get_transforms(train=False),
    )

    train_dataset = _maybe_subset(train_dataset, subset_fraction, seed)
    val_dataset = _maybe_subset(val_dataset, subset_fraction, seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,  # only useful with a GPU
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )
