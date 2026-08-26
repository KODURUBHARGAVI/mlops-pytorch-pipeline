"""The web API that serves predictions from the trained model.

It has three endpoints: /health, /metadata and /predict.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    CLASSES,
    DATASET_NAME,
    IMAGE_SIZE,
    IN_CHANNELS,
    get_inference_transform,
)
from model import get_model  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("serve")

CONFIG_SEARCH_PATHS = (
    "/app/configs/training_config.yaml",
    "configs/training_config.yaml",
)
DEFAULT_CHECKPOINT = "/app/checkpoints/classifier_v1.pt"
TOP_K = int(os.getenv("TOP_K", "5"))

# Filled in when the app starts up.
STATE: dict = {
    "model": None,
    "classes": list(CLASSES),
    "transform": get_inference_transform(),
    "metadata": {},
}


def log(**fields) -> None:
    logger.info(json.dumps(fields))


def resolve_checkpoint_path() -> Path:
    """Locate the checkpoint: $MODEL_PATH, then the config, then the default."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        return Path(env_path)

    for candidate in CONFIG_SEARCH_PATHS:
        config_file = Path(candidate)
        if config_file.exists():
            try:
                config = yaml.safe_load(config_file.read_text())
                output = config["output"]
                # CHECKPOINT_DIR overrides the config, matching train.py.
                directory = os.getenv("CHECKPOINT_DIR", output["checkpoint_dir"])
                return Path(directory) / output["model_name"]
            except (KeyError, TypeError, yaml.YAMLError):
                log(event="config_unreadable", path=str(config_file))
    return Path(DEFAULT_CHECKPOINT)


def load_model() -> None:
    """Load the checkpoint into STATE. Never raises: /health reports failure."""
    checkpoint_path = resolve_checkpoint_path()
    if not checkpoint_path.exists():
        log(event="checkpoint_missing", path=str(checkpoint_path))
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        architecture = checkpoint.get("architecture", "simple_cnn")
        classes = checkpoint.get("classes", list(CLASSES))
        num_classes = int(checkpoint.get("num_classes", len(classes)))
        in_channels = int(checkpoint.get("in_channels", IN_CHANNELS))

        model = get_model(
            architecture=architecture,
            num_classes=num_classes,
            in_channels=in_channels,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        STATE["model"] = model
        STATE["classes"] = classes
        STATE["transform"] = get_inference_transform()
        STATE["metadata"] = {
            "architecture": architecture,
            "dataset": checkpoint.get("dataset", DATASET_NAME),
            "input_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}x{in_channels}",
            "num_classes": num_classes,
            "checkpoint": str(checkpoint_path),
            "trained_epochs": checkpoint.get("epoch"),
            "val_accuracy": checkpoint.get("val_accuracy"),
            "val_loss": checkpoint.get("val_loss"),
        }
        log(event="model_loaded", **STATE["metadata"])
    except Exception as exc:  # noqa: BLE001 - surfaced through /health
        log(event="model_load_failed", path=str(checkpoint_path), error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))
    load_model()
    yield
    STATE["model"] = None


app = FastAPI(
    title="MLOps PyTorch Pipeline - Inference API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> JSONResponse:
    """200 only when a model is loaded, so k8s can gate traffic on it."""
    if STATE["model"] is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "model not loaded"},
        )
    return JSONResponse(status_code=200, content={"status": "healthy", "model_loaded": True})


@app.get("/metadata")
def metadata() -> dict:
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": STATE["classes"], **STATE["metadata"]}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    """Classify one uploaded image and return class probabilities."""
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        pil_image = Image.open(io.BytesIO(raw))
        # Force the decode here so a corrupt file becomes a 400, not a 500
        # later. The transform handles the colour-mode conversion.
        pil_image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail="File is not a readable image"
        ) from exc

    started = time.perf_counter()
    tensor = STATE["transform"](pil_image).unsqueeze(0)
    with torch.no_grad():
        probabilities = F.softmax(STATE["model"](tensor), dim=1)[0]
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    classes = STATE["classes"]
    k = min(TOP_K, len(classes))
    top_probs, top_indices = torch.topk(probabilities, k)

    response = {
        "filename": image.filename,
        "predicted_class": classes[int(top_indices[0])],
        "confidence": round(float(top_probs[0]), 4),
        "top_k": [
            {"class": classes[int(i)], "probability": round(float(p), 4)}
            for p, i in zip(top_probs, top_indices, strict=True)
        ],
        "probabilities": {
            # strict=False: a checkpoint whose class list disagrees with the head
            # width should degrade, not 500 on every request.
            cls: round(float(prob), 4)
            for cls, prob in zip(classes, probabilities, strict=False)
        },
        "latency_ms": latency_ms,
    }
    log(
        event="prediction",
        filename=image.filename,
        predicted_class=response["predicted_class"],
        confidence=response["confidence"],
        latency_ms=latency_ms,
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host="0.0.0.0",  # noqa: S104 - container needs to bind all interfaces
        port=int(os.getenv("PORT", "8080")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
