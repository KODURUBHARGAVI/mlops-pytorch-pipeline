"""Tests for the web API."""

import io

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from dataset import CLASSES, IN_CHANNELS, NUM_CLASSES  # noqa: E402
from model import get_model  # noqa: E402


@pytest.fixture()
def checkpoint(tmp_path):
    """A small but real saved model that the API can load."""
    model = get_model("simple_cnn", NUM_CLASSES, in_channels=IN_CHANNELS)
    path = tmp_path / "classifier_v1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "simple_cnn",
            "num_classes": NUM_CLASSES,
            "in_channels": IN_CHANNELS,
            "classes": list(CLASSES),
            "epoch": 1,
            "val_accuracy": 0.5,
            "val_loss": 1.2,
        },
        path,
    )
    return path


def build_client(monkeypatch, model_path):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    import serve

    serve.STATE["model"] = None
    return TestClient(serve.app)


def png_bytes(size=(64, 64), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    colour = (10, 200, 90) if mode == "RGB" else 128
    Image.new(mode, size, color=colour).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_returns_503_when_there_is_no_model(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path / "missing.pt") as client:
        assert client.get("/health").status_code == 503


def test_health_returns_200_once_the_model_is_loaded(monkeypatch, checkpoint):
    with build_client(monkeypatch, checkpoint) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_metadata_describes_the_loaded_model(monkeypatch, checkpoint):
    with build_client(monkeypatch, checkpoint) as client:
        body = client.get("/metadata").json()
        assert body["architecture"] == "simple_cnn"
        assert body["classes"] == list(CLASSES)
        assert body["val_accuracy"] == 0.5


def test_predict_returns_a_score_for_every_class(monkeypatch, checkpoint):
    with build_client(monkeypatch, checkpoint) as client:
        response = client.post(
            "/predict", files={"image": ("test_image.png", png_bytes(), "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_class"] in body["probabilities"]
        assert len(body["probabilities"]) == NUM_CLASSES
        assert pytest.approx(sum(body["probabilities"].values()), abs=0.01) == 1.0


def test_predict_accepts_odd_sizes_and_grey_images(monkeypatch, checkpoint):
    with build_client(monkeypatch, checkpoint) as client:
        for size, mode in [((17, 400), "RGB"), ((300, 220), "L")]:
            response = client.post(
                "/predict",
                files={"image": ("odd.png", png_bytes(size, mode), "image/png")},
            )
            assert response.status_code == 200


def test_predict_rejects_a_file_that_is_not_an_image(monkeypatch, checkpoint):
    with build_client(monkeypatch, checkpoint) as client:
        response = client.post(
            "/predict", files={"image": ("notes.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
