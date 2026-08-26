"""Unit tests for the model, the data transforms and the training configs.

Everything here runs on CPU in a few seconds and touches no network, which is
what makes it safe to gate every pull request on.
"""

from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

import train
from dataset import DATASET_SPECS, get_inference_transform, get_spec, get_transforms
from model import SUPPORTED_ARCHITECTURES, SimpleCNN, get_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
ALL_CONFIGS = sorted(CONFIG_DIR.glob("training_config*.yaml"))
COMBINATIONS = [(a, d) for a in SUPPORTED_ARCHITECTURES for d in DATASET_SPECS]


def blank(spec, size=None):
    """A PIL image matching a dataset's colour mode, at an arbitrary size."""
    mode = "L" if spec.in_channels == 1 else "RGB"
    return Image.new(mode, size or (spec.image_size, spec.image_size))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("architecture", "dataset"), COMBINATIONS)
def test_forward_pass_returns_logits_per_class(architecture, dataset):
    """Every architecture must handle every dataset's shape and channel count."""
    spec = get_spec(dataset)
    model = get_model(
        architecture=architecture,
        num_classes=spec.num_classes,
        in_channels=spec.in_channels,
    )
    model.eval()
    batch = torch.randn(2, spec.in_channels, spec.image_size, spec.image_size)
    with torch.no_grad():
        output = model(batch)
    assert output.shape == (2, spec.num_classes)
    assert torch.isfinite(output).all()


def test_num_classes_is_honoured():
    model = get_model(architecture="simple_cnn", num_classes=4)
    with torch.no_grad():
        assert model(torch.randn(1, 3, 32, 32)).shape == (1, 4)


def test_grayscale_model_rejects_rgb_input():
    """A 1-channel model must fail loudly on 3-channel input, not silently."""
    model = get_model("simple_cnn", num_classes=10, in_channels=1)
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 3, 28, 28))


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_model(architecture="not-a-real-model")


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unsupported dataset"):
        get_spec("imagenet")


def test_dataset_name_lookup_is_forgiving():
    assert get_spec("Fashion-MNIST").name == "fashionmnist"
    assert get_spec("fashion_mnist").name == "fashionmnist"


def test_model_is_trainable():
    """One backward pass must produce gradients on the trainable parameters."""
    model = SimpleCNN(num_classes=10)
    loss = torch.nn.functional.cross_entropy(
        model(torch.randn(2, 3, 32, 32)), torch.tensor([1, 7])
    )
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


@pytest.mark.parametrize("dataset", list(DATASET_SPECS))
def test_checkpoint_roundtrip(tmp_path, dataset):
    """A checkpoint must carry enough metadata for serve.py to rebuild the model."""
    spec = get_spec(dataset)
    model = get_model("simple_cnn", spec.num_classes, in_channels=spec.in_channels)
    checkpoint_path = tmp_path / "classifier_v1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "simple_cnn",
            "dataset": spec.name,
            "num_classes": spec.num_classes,
            "in_channels": spec.in_channels,
            "classes": list(spec.classes),
        },
        checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    restored = get_model(
        checkpoint["architecture"],
        checkpoint["num_classes"],
        in_channels=checkpoint["in_channels"],
    )
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()
    model.eval()

    sample = torch.randn(1, spec.in_channels, spec.image_size, spec.image_size)
    with torch.no_grad():
        assert torch.allclose(model(sample), restored(sample), atol=1e-6)


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", list(DATASET_SPECS))
def test_training_transforms_yield_the_datasets_native_shape(dataset):
    spec = get_spec(dataset)
    expected = (spec.in_channels, spec.image_size, spec.image_size)
    assert get_transforms(dataset, train=True)(blank(spec)).shape == expected
    assert get_transforms(dataset, train=False)(blank(spec)).shape == expected


UPLOAD_MODES = ["RGB", "L", "RGBA", "P"]


@pytest.mark.parametrize("dataset", list(DATASET_SPECS))
@pytest.mark.parametrize("mode", UPLOAD_MODES)
def test_inference_transform_normalises_any_upload(dataset, mode):
    """Uploads arrive at any size and colour mode; serving must cope with all of them.

    Regression test: a grayscale upload used to reach a 3-channel Normalize
    and raise a broadcast error mid-request.
    """
    spec = get_spec(dataset)
    transform = get_inference_transform(dataset)
    expected = (spec.in_channels, spec.image_size, spec.image_size)
    for size in [(200, 130), (17, 400), (spec.image_size, spec.image_size)]:
        tensor = transform(Image.new(mode, size))
        assert tensor.shape == expected
        assert torch.isfinite(tensor).all()


@pytest.mark.parametrize("dataset", list(DATASET_SPECS))
def test_normalisation_stats_match_channel_count(dataset):
    spec = get_spec(dataset)
    assert len(spec.mean) == len(spec.std) == spec.in_channels


# --------------------------------------------------------------------------
# Configs
# --------------------------------------------------------------------------


def test_env_overrides_change_the_config(monkeypatch):
    """Env vars let a quick run reuse the same config file."""
    config = yaml.safe_load((CONFIG_DIR / "training_config.yaml").read_text())
    monkeypatch.setenv("EPOCHS", "2")
    monkeypatch.setenv("SUBSET_FRACTION", "0.15")
    monkeypatch.setenv("DATA_DIR", "/mounted/data")
    monkeypatch.setenv("CHECKPOINT_DIR", "/mounted/checkpoints")

    applied = train.apply_env_overrides(config)

    assert config["training"]["epochs"] == 2
    assert config["data"]["subset_fraction"] == 0.15
    assert config["data"]["data_dir"] == "/mounted/data"
    assert config["output"]["checkpoint_dir"] == "/mounted/checkpoints"
    assert set(applied) == {"EPOCHS", "SUBSET_FRACTION", "DATA_DIR", "CHECKPOINT_DIR"}


def test_config_is_untouched_when_no_env_vars_are_set(monkeypatch):
    for name in train.ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    original = yaml.safe_load((CONFIG_DIR / "training_config.yaml").read_text())
    config = yaml.safe_load((CONFIG_DIR / "training_config.yaml").read_text())

    assert train.apply_env_overrides(config) == {}
    assert config == original


@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=lambda p: p.name)
def test_every_config_is_valid(config_path):
    """Each shipped config must be internally consistent and runnable."""
    config = yaml.safe_load(config_path.read_text())
    spec = get_spec(config["data"]["dataset"])

    assert config["model"]["architecture"] in SUPPORTED_ARCHITECTURES
    assert config["model"]["num_classes"] == spec.num_classes
    assert config["training"]["epochs"] > 0
    assert config["training"]["batch_size"] > 0
    assert config["training"]["learning_rate"] > 0
    assert config["training"]["early_stopping_patience"] >= 1
    assert 0 < config["data"].get("subset_fraction", 1.0) <= 1.0
    assert config["output"]["model_name"].endswith(".pt")
