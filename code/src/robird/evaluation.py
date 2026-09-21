from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from robird.data import ObservationBatch
from robird.metrics import budget_curve
from robird.models import ObservationClassifier


@torch.no_grad()
def evaluate_loader(
    model: ObservationClassifier,
    loader: DataLoader[ObservationBatch],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    model.to(device)
    records: list[dict[str, Any]] = []
    for batch in loader:
        features = batch.features.to(device, non_blocking=True)
        mask = batch.mask.to(device, non_blocking=True)
        output = model(features, mask)
        probabilities = torch.softmax(output.logits, dim=1).cpu().numpy()
        weights = output.view_weights.cpu().numpy() if output.view_weights is not None else None
        for position in range(len(batch.labels)):
            record: dict[str, Any] = {
                "observation_id": int(batch.observation_ids[position]),
                "budget": int(batch.budgets[position]),
                "label": int(batch.labels[position]),
                "photo_ids": list(batch.photo_ids[position]),
                "probabilities": probabilities[position],
                "prediction": int(np.argmax(probabilities[position])),
            }
            if weights is not None:
                valid_views = int(batch.mask[position].sum())
                record["view_weights"] = weights[position, :valid_views]
            records.append(record)
    return records


def summarize_records(
    records: list[dict[str, Any]], ece_bins: int, topk: Sequence[int] = (1, 5)
) -> dict[str, Any]:
    if not records:
        raise ValueError("No evaluation records")
    return budget_curve(records, ece_bins=ece_bins, topk=topk)
