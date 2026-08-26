"""API-level tests for the FastAPI inference service.

These exercise the real app through a test client, including the checkpoint
loading that the Kubernetes readiness probe depends on.
"""

import io

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from dataset import DATASET_SPECS, get_spec  # noqa: E402
from model import get_model  # noqa: E402


def write_checkpoint(tmp_path, dataset="fashionmnist"):
    """A real, tiny checkpoint that serve.py can actually load."""
    spec = get_spec(dataset)
    model = get_model("simple_cnn", spec.num_classes, in_channels=spec.in_channels)
    path = tmp_path / "classifier_v1.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "simple_cnn",
            "dataset": spec.name,
            "num_classes": spec.num_classes,
            "in_channels": spec.in_channels,
            "classes": list(spec.classes),
            "epoch": 1,
            "val_accuracy": 0.5,
            "val_loss": 1.2,
        },
        path,
    )
    return path


@pytest.fixture()
def checkpoint(tmp_path):
    return write_checkpoint(tmp_path)


def build_client(monkeypatch, model_path):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    import serve

    serve.STATE["model"] = None
    return TestClient(serve.app)


def png_bytes(size=(64, 64), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=(10, 200, 90) if mode == "RGB" else 128).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


def test_health_is_503_without_a_checkpoint(monkeypatch, tmp_path):
    with build_client(monkeypatch, tmp_path / "missing.pt") as client:
        assert client.get("/health").status_code == 503


def test_health_is_200_when_model_loaded(monkeypatch, checkpoint):
    with build_client(monkeypatch, checkpoint) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


@pytest.mark.parametrize("dataset", list(DATASET_SPECS))
def test_predict_works_for_every_dataset(monkeypatch, tmp_path, dataset):
    """The image serves whichever dataset the checkpoint was trained on."""
    path = write_checkpoint(tmp_path, dataset)
    with build_client(monkeypatch, path) as client:
        response = client.post(
            "/predict", files={"image": ("test_image.png", png_bytes(), "image/png")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_class"] in body["probabilities"]
        assert len(body["probabilities"]) == 10
        assert pytest.approx(sum(body["probabilities"].values()), abs=0.01) == 1.0
        assert client.get("/metadata").json()["dataset"] == get_spec(dataset).name


def test_predict_accepts_odd_sizes_and_grayscale_uploads(monkeypatch, checkpoint):
    with build_client(monkeypatch, checkpoint) as client:
        for size, mode in [((17, 400), "RGB"), ((300, 220), "L")]:
            response = client.post(
                "/predict",
                files={"image": ("odd.png", png_bytes(size, mode), "image/png")},
            )
            assert response.status_code == 200


def test_predict_rejects_a_non_image(monkeypatch, checkpoint):
    with build_client(monkeypatch, checkpoint) as client:
        response = client.post(
            "/predict", files={"image": ("notes.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
