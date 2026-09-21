from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from robird.data import ObservationBatch
from robird.metrics import macro_top1, negative_log_likelihood
from robird.models import ObservationClassifier


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(batch: ObservationBatch, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch.features.to(device, non_blocking=True),
        batch.mask.to(device, non_blocking=True),
        batch.labels.to(device, non_blocking=True),
    )


def train_one_epoch(
    model: ObservationClassifier,
    loader: DataLoader[ObservationBatch],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        features, mask, labels = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(features, mask)
        loss = criterion(output.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        total_items += len(labels)
    if total_items == 0:
        raise RuntimeError("Training loader produced no observations")
    return total_loss / total_items


@torch.no_grad()
def validate(
    model: ObservationClassifier,
    loader: DataLoader[ObservationBatch],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    grouped_probabilities: dict[int, list[np.ndarray]] = {}
    grouped_labels: dict[int, int] = {}
    for batch in loader:
        features, mask, labels = _move(batch, device)
        output = model(features, mask)
        batch_probabilities = torch.softmax(output.logits, dim=1).cpu().numpy()
        for position, observation_id_tensor in enumerate(batch.observation_ids):
            observation_id = int(observation_id_tensor)
            label = int(batch.labels[position])
            previous_label = grouped_labels.setdefault(observation_id, label)
            if previous_label != label:
                raise ValueError(f"Validation label mismatch for observation {observation_id}")
            grouped_probabilities.setdefault(observation_id, []).append(
                batch_probabilities[position]
            )
    if not grouped_probabilities:
        raise RuntimeError("Validation loader produced no observations")
    observation_ids = sorted(grouped_probabilities)
    probabilities_array = np.stack(
        [np.mean(grouped_probabilities[value], axis=0) for value in observation_ids]
    )
    labels_array = np.asarray([grouped_labels[value] for value in observation_ids], dtype=np.int64)
    return {
        "loss": negative_log_likelihood(probabilities_array, labels_array),
        "macro_top1": macro_top1(probabilities_array, labels_array),
        "micro_top1": float(np.mean(probabilities_array.argmax(axis=1) == labels_array)),
    }


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def fit(
    model: ObservationClassifier,
    train_loader: DataLoader[ObservationBatch],
    validation_loader: DataLoader[ObservationBatch],
    config: Mapping[str, Any],
    device: torch.device,
    checkpoint_path: Path,
    provenance: Mapping[str, Any],
) -> list[dict[str, float]]:
    training = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training["label_smoothing"]))
    model.to(device)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(int(training["epochs"])):
        dataset = getattr(train_loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        validation = validate(model, validation_loader, device)
        row = {"epoch": float(epoch), "train_loss": train_loss, **validation}
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "provenance": dict(provenance),
                },
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["patience"]):
                break
    return history
