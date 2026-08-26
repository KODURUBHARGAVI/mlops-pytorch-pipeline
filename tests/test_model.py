"""Tests for the model, the image preparation and the settings file."""

from pathlib import Path

import pytest
import torch
import yaml
from PIL import Image

import train
from dataset import (
    CLASSES,
    IMAGE_MODE,
    IMAGE_SIZE,
    IN_CHANNELS,
    NUM_CLASSES,
    get_inference_transform,
    get_transforms,
)
from model import SUPPORTED_ARCHITECTURES, SimpleCNN, get_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"

SHAPE = (IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("architecture", SUPPORTED_ARCHITECTURES)
def test_forward_pass_returns_one_score_per_class(architecture):
    """Two images in should give two rows of 10 scores out."""
    model = get_model(architecture, num_classes=NUM_CLASSES, in_channels=IN_CHANNELS)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(2, *SHAPE))
    assert output.shape == (2, NUM_CLASSES)
    assert torch.isfinite(output).all()


def test_number_of_classes_can_be_changed():
    model = get_model("simple_cnn", num_classes=4, in_channels=IN_CHANNELS)
    with torch.no_grad():
        assert model(torch.randn(1, *SHAPE)).shape == (1, 4)


def test_model_rejects_the_wrong_number_of_channels():
    """A grey model given a colour image should fail clearly, not carry on."""
    model = get_model("simple_cnn", num_classes=NUM_CLASSES, in_channels=1)
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        get_model(architecture="not-a-real-model")


def test_model_can_learn():
    """After one backward pass the weights should have gradients."""
    model = SimpleCNN(num_classes=NUM_CLASSES, in_channels=IN_CHANNELS)
    loss = torch.nn.functional.cross_entropy(
        model(torch.randn(2, *SHAPE)), torch.tensor([1, 7])
    )
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_saved_model_loads_back_the_same(tmp_path):
    """A saved file must hold enough detail for the API to rebuild the model."""
    model = get_model("simple_cnn", NUM_CLASSES, in_channels=IN_CHANNELS)
    checkpoint_path = tmp_path / "classifier_v1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "simple_cnn",
            "num_classes": NUM_CLASSES,
            "in_channels": IN_CHANNELS,
            "classes": list(CLASSES),
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

    sample = torch.randn(1, *SHAPE)
    with torch.no_grad():
        assert torch.allclose(model(sample), restored(sample), atol=1e-6)


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------


def test_training_steps_give_the_expected_shape():
    image = Image.new(IMAGE_MODE, (IMAGE_SIZE, IMAGE_SIZE))
    assert get_transforms(train=True)(image).shape == SHAPE
    assert get_transforms(train=False)(image).shape == SHAPE


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
@pytest.mark.parametrize("size", [(200, 130), (17, 400), (IMAGE_SIZE, IMAGE_SIZE)])
def test_uploaded_images_are_always_converted_correctly(mode, size):
    """An upload can be any size or colour mode and must still work."""
    tensor = get_inference_transform()(Image.new(mode, size))
    assert tensor.shape == SHAPE
    assert torch.isfinite(tensor).all()


def test_there_are_ten_class_names():
    assert len(CLASSES) == NUM_CLASSES == 10


# --------------------------------------------------------------------------
# Settings file
# --------------------------------------------------------------------------


def test_settings_file_is_valid():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    assert config["model"]["architecture"] in SUPPORTED_ARCHITECTURES
    assert config["model"]["num_classes"] == NUM_CLASSES
    assert config["training"]["epochs"] > 0
    assert config["training"]["batch_size"] > 0
    assert config["training"]["learning_rate"] > 0
    assert config["training"]["early_stopping_patience"] >= 1
    assert 0 < config["data"]["subset_fraction"] <= 1.0
    assert config["output"]["model_name"].endswith(".pt")


def test_environment_variables_change_the_settings(monkeypatch):
    config = yaml.safe_load(CONFIG_PATH.read_text())
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


def test_settings_stay_the_same_when_no_variables_are_set(monkeypatch):
    for name in train.ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    original = yaml.safe_load(CONFIG_PATH.read_text())
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert train.apply_env_overrides(config) == {}
    assert config == original
