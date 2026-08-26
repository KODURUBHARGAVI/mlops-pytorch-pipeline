"""Dataset registry, transforms and DataLoaders.

Two datasets are supported, selected purely from config (`data.dataset`):

* ``fashionmnist`` - 28x28 grayscale, ~30 MB, fast to fetch and fast to train.
* ``cifar10``      - 32x32 RGB, ~163 MB from a notoriously slow origin server.

Both expose the same interface, so ``train.py`` and ``serve.py`` never branch
on the dataset name. The spec describing each one (image size, channel count,
normalisation constants, class names) is the single source of truth shared by
training and inference, which stops the two pipelines from drifting apart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

__all__ = [
    "ConvertImageMode",
    "DatasetSpec",
    "DATASET_SPECS",
    "CIFAR10_CLASSES",
    "FASHION_MNIST_CLASSES",
    "get_spec",
    "get_transforms",
    "get_inference_transform",
    "get_dataloaders",
]

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

FASHION_MNIST_CLASSES = (
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


class ConvertImageMode:
    """Force a PIL image into the colour mode the model was trained on.

    Uploads arrive as RGB, grayscale, palette or RGBA depending on what the
    caller had lying around. Without this, a 1-channel upload reaches a
    3-channel Normalize (or vice versa) and blows up mid-request. A named
    class rather than a Lambda so the transform stays picklable and printable.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(self, image):
        return image if image.mode == self.mode else image.convert(self.mode)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mode={self.mode!r})"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything training and serving need to know about one dataset."""

    name: str
    torchvision_class: type
    image_size: int
    in_channels: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    classes: tuple[str, ...]
    crop_padding: int
    horizontal_flip: bool = True
    approx_download_mb: int = 0
    extra_kwargs: dict = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def image_mode(self) -> str:
        """The PIL mode matching this dataset's channel count."""
        return "L" if self.in_channels == 1 else "RGB"


DATASET_SPECS: dict[str, DatasetSpec] = {
    "cifar10": DatasetSpec(
        name="cifar10",
        torchvision_class=datasets.CIFAR10,
        image_size=32,
        in_channels=3,
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        classes=CIFAR10_CLASSES,
        crop_padding=4,
        horizontal_flip=True,
        approx_download_mb=163,
    ),
    "fashionmnist": DatasetSpec(
        name="fashionmnist",
        torchvision_class=datasets.FashionMNIST,
        image_size=28,
        in_channels=1,
        mean=(0.2860,),
        std=(0.3530,),
        classes=FASHION_MNIST_CLASSES,
        crop_padding=2,
        horizontal_flip=True,
        approx_download_mb=30,
    ),
}


def get_spec(dataset: str = "cifar10") -> DatasetSpec:
    """Look up a dataset spec by name, tolerating hyphens and underscores."""
    key = dataset.lower().strip().replace("-", "").replace("_", "")
    if key not in DATASET_SPECS:
        raise ValueError(
            f"Unsupported dataset {dataset!r}. Expected one of: {', '.join(DATASET_SPECS)}"
        )
    return DATASET_SPECS[key]


def get_transforms(dataset: str = "cifar10", train: bool = True) -> transforms.Compose:
    """Augmentation + normalisation pipeline. Augmentation is training-only."""
    spec = get_spec(dataset)
    steps: list = []
    if train:
        if spec.horizontal_flip:
            steps.append(transforms.RandomHorizontalFlip())
        steps.append(transforms.RandomCrop(spec.image_size, padding=spec.crop_padding))
    steps += [transforms.ToTensor(), transforms.Normalize(spec.mean, spec.std)]
    return transforms.Compose(steps)


def get_inference_transform(dataset: str = "cifar10") -> transforms.Compose:
    """Pre-processing for one uploaded image of arbitrary size and colour mode.

    Colour-mode coercion comes first: everything downstream then sees exactly
    ``in_channels`` channels, whatever the caller uploaded.
    """
    spec = get_spec(dataset)
    return transforms.Compose(
        [
            ConvertImageMode(spec.image_mode),
            transforms.Resize((spec.image_size, spec.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ]
    )


def _maybe_subset(dataset, fraction: float, seed: int):
    """Deterministically keep `fraction` of a dataset (1.0 = everything)."""
    if fraction >= 1.0:
        return dataset
    if not 0 < fraction < 1:
        raise ValueError(f"subset_fraction must be in (0, 1], got {fraction}")
    keep = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:keep].tolist()
    return Subset(dataset, indices)


def get_dataloaders(
    data_dir: str,
    dataset: str = "cifar10",
    batch_size: int = 64,
    num_workers: int | None = None,
    download: bool = True,
    subset_fraction: float = 1.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build the train/validation DataLoaders.

    Args:
        data_dir: Where the dataset lives. Inside a container this is a mounted
            volume, so the download survives restarts and happens exactly once.
        dataset: ``cifar10`` or ``fashionmnist``.
        batch_size: Mini-batch size for both loaders.
        num_workers: Worker processes; defaults to ``$DATALOADER_WORKERS`` or 2.
        download: Fetch the dataset if it is not already on disk. Pre-seed
            ``data_dir`` with ``scripts/fetch_data.py`` to skip the slow origin.
        subset_fraction: Fraction of each split to keep. Small values give a
            fast smoke test; 1.0 is a real run.
        seed: Seeds the subset sampling so runs stay reproducible.
    """
    spec = get_spec(dataset)
    if num_workers is None:
        num_workers = int(os.getenv("DATALOADER_WORKERS", "2"))

    os.makedirs(data_dir, exist_ok=True)

    train_dataset = spec.torchvision_class(
        root=data_dir,
        train=True,
        download=download,
        transform=get_transforms(spec.name, train=True),
        **spec.extra_kwargs,
    )
    val_dataset = spec.torchvision_class(
        root=data_dir,
        train=False,
        download=download,
        transform=get_transforms(spec.name, train=False),
        **spec.extra_kwargs,
    )

    train_dataset = _maybe_subset(train_dataset, subset_fraction, seed)
    val_dataset = _maybe_subset(val_dataset, subset_fraction, seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,  # CPU-only training: pinning buys nothing.
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
    )
