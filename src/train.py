"""Trains the classifier and saves the best model.

Settings are read from configs/training_config.yaml. Progress is printed as
one JSON object per line.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

# Support both `python src/train.py` and `python -m src.train`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    CLASSES,
    DATASET_NAME,
    IN_CHANNELS,
    NUM_CLASSES,
    get_dataloaders,
)
from model import get_model  # noqa: E402

CONFIG_SEARCH_PATHS = (
    "/app/configs/training_config.yaml",
    "configs/training_config.yaml",
)

# Environment variables that override a config value, so a quick run does not
# need its own config file. Maps: env var -> (section, key, type).
ENV_OVERRIDES = {
    "DATA_DIR": ("data", "data_dir", str),
    "CHECKPOINT_DIR": ("output", "checkpoint_dir", str),
    "EPOCHS": ("training", "epochs", int),
    "SUBSET_FRACTION": ("data", "subset_fraction", float),
}


def log(**fields) -> None:
    """Emit a single structured JSON line to stdout."""
    print(json.dumps(fields), flush=True)


def resolve_config_path() -> Path:
    """Find the training config, preferring the mounted ConfigMap."""
    env_path = os.getenv("CONFIG_PATH")
    candidates = ([env_path] if env_path else []) + list(CONFIG_SEARCH_PATHS)
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "No training config found. Looked in: " + ", ".join(candidates)
    )


def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_env_overrides(config: dict) -> dict:
    """Let environment variables override config values. Returns what changed."""
    applied = {}
    for env_name, (section, key, cast) in ENV_OVERRIDES.items():
        raw = os.getenv(env_name)
        if not raw:
            continue
        config[section][key] = cast(raw)
        applied[env_name] = config[section][key]
    return applied


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


def main() -> None:
    config_path = resolve_config_path()
    config = load_config(config_path)
    overrides = apply_env_overrides(config)

    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    output_cfg = config["output"]

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    num_classes = int(model_cfg.get("num_classes", NUM_CLASSES))
    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"the settings file says num_classes={num_classes}, but the dataset "
            f"has {NUM_CLASSES} classes"
        )
    subset_fraction = float(data_cfg.get("subset_fraction", 1.0))

    # CPU-only by design; the code still picks up a GPU if one is present.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", os.cpu_count() or 1)))

    log(
        event="run_started",
        config_path=str(config_path),
        device=str(device),
        threads=torch.get_num_threads(),
        dataset=DATASET_NAME,
        architecture=model_cfg["architecture"],
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        subset_fraction=subset_fraction,
        env_overrides=overrides or None,
    )

    model = get_model(
        architecture=model_cfg["architecture"],
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", False)),
        in_channels=IN_CHANNELS,
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=data_cfg["data_dir"],
        batch_size=int(train_cfg["batch_size"]),
        num_workers=data_cfg.get("num_workers"),
        download=bool(data_cfg.get("download", True)),
        subset_fraction=subset_fraction,
        seed=seed,
    )
    log(
        event="data_ready",
        dataset=DATASET_NAME,
        train_batches=len(train_loader),
        val_batches=len(val_loader),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(output_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / output_cfg["model_name"]

    patience = int(train_cfg["early_stopping_patience"])
    best_val_loss, best_val_acc, patience_counter = float("inf"), 0.0, 0

    for epoch in range(int(train_cfg["epochs"])):
        started = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log(
            event="epoch_completed",
            epoch=epoch + 1,
            train_loss=round(train_loss, 4),
            train_accuracy=round(train_acc, 4),
            val_loss=round(val_loss, 4),
            val_accuracy=round(val_acc, 4),
            duration_seconds=round(time.time() - started, 1),
        )

        if val_loss < best_val_loss:
            best_val_loss, best_val_acc, patience_counter = val_loss, val_acc, 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    # Everything serve.py needs to rebuild the model:
                    "architecture": model_cfg["architecture"],
                    "num_classes": num_classes,
                    "in_channels": IN_CHANNELS,
                    "dataset": DATASET_NAME,
                    "classes": list(CLASSES),
                },
                save_path,
            )
            log(event="checkpoint_saved", path=str(save_path), epoch=epoch + 1)
        else:
            patience_counter += 1
            log(
                event="no_improvement",
                epoch=epoch + 1,
                patience_counter=patience_counter,
                patience=patience,
            )
            if patience_counter >= patience:
                log(event="early_stopping", epoch=epoch + 1)
                break

    log(
        event="training_complete",
        best_val_loss=round(best_val_loss, 4),
        best_val_accuracy=round(best_val_acc, 4),
        checkpoint=str(save_path),
    )


if __name__ == "__main__":
    main()
