from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from robird.budgets import enumerate_budget_subsets, stable_key


@dataclass(frozen=True)
class ObservationBatch:
    features: Tensor
    mask: Tensor
    labels: Tensor
    observation_ids: Tensor
    budgets: Tensor
    photo_ids: list[list[int]]


def load_feature_table(matrix_path: Path, index_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Feature matrix must be non-empty [N,D], got {matrix.shape}")
    index = pd.read_csv(index_path)
    if "photo_id" not in index:
        raise ValueError("Feature index requires photo_id")
    if index["photo_id"].duplicated().any():
        raise ValueError("Feature index photo_id is not unique")
    if len(index) != matrix.shape[0]:
        raise ValueError("Feature index length differs from feature matrix rows")
    if "feature_row" not in index:
        index = index.copy()
        index["feature_row"] = np.arange(len(index), dtype=np.int64)
    rows = index["feature_row"].astype(np.int64)
    if rows.duplicated().any() or (rows < 0).any() or (rows >= matrix.shape[0]).any():
        raise ValueError("feature_row must be unique and within the feature matrix")
    index = index.copy()
    index["photo_id"] = index["photo_id"].astype(np.int64)
    index["feature_row"] = rows
    return matrix, index


class ObservationFeatureDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        features: np.ndarray,
        feature_index: pd.DataFrame,
        split: str,
        budget: int | None,
        seed: int,
        subset_mode: str,
        max_subsets: int,
    ) -> None:
        if subset_mode not in {"random", "exhaustive", "first"}:
            raise ValueError(f"Unknown subset_mode: {subset_mode}")
        self.features = features
        self.seed = int(seed)
        self.subset_mode = subset_mode
        self.max_subsets = int(max_subsets)
        self.epoch = 0
        feature_rows = {
            int(row.photo_id): int(row.feature_row) for row in feature_index.itertuples(index=False)
        }
        selected = manifest.loc[manifest["split"] == split].copy()
        missing = sorted(set(int(v) for v in selected["photo_id"]) - set(feature_rows))
        if missing:
            raise ValueError(f"Feature rows missing for photo IDs: {missing[:10]}")
        self.groups: dict[int, dict[str, Any]] = {}
        for observation_id, group in selected.groupby("observation_id", sort=True):
            labels = group["class_index"].unique()
            if len(labels) != 1:
                raise ValueError(f"Observation {observation_id} has multiple class indices")
            photo_ids = sorted(int(value) for value in group["photo_id"])
            use_budget = len(photo_ids) if budget is None else int(budget)
            if use_budget > len(photo_ids):
                continue
            subsets = enumerate_budget_subsets(
                photo_ids, use_budget, self.max_subsets, self.seed, int(observation_id)
            )
            self.groups[int(observation_id)] = {
                "label": int(labels[0]),
                "budget": use_budget,
                "subsets": subsets,
                "feature_rows": {photo_id: feature_rows[photo_id] for photo_id in photo_ids},
            }
        self.samples: list[tuple[int, int]] = []
        for observation_id, group in self.groups.items():
            if subset_mode == "exhaustive":
                self.samples.extend((observation_id, index) for index in range(len(group["subsets"])))
            else:
                self.samples.append((observation_id, 0))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        observation_id, subset_index = self.samples[index]
        group = self.groups[observation_id]
        subsets = group["subsets"]
        if self.subset_mode == "random":
            subset_index = stable_key(self.seed, self.epoch, observation_id) % len(subsets)
        elif self.subset_mode == "first":
            subset_index = 0
        subset = subsets[subset_index]
        rows = [group["feature_rows"][photo_id] for photo_id in subset]
        feature_values = np.asarray(self.features[rows], dtype=np.float32)
        return {
            "features": torch.from_numpy(feature_values.copy()),
            "label": int(group["label"]),
            "observation_id": observation_id,
            "budget": int(group["budget"]),
            "photo_ids": list(subset),
        }


def collate_observations(items: list[dict[str, Any]]) -> ObservationBatch:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    max_views = max(int(item["features"].shape[0]) for item in items)
    feature_dim = int(items[0]["features"].shape[1])
    features = torch.zeros((len(items), max_views, feature_dim), dtype=torch.float32)
    mask = torch.zeros((len(items), max_views), dtype=torch.bool)
    for position, item in enumerate(items):
        values = item["features"]
        if values.ndim != 2 or int(values.shape[0]) < 1:
            raise ValueError("Each item must contain a non-empty [V,D] feature tensor")
        if int(values.shape[1]) != feature_dim:
            raise ValueError("Feature dimensions differ within batch")
        views = int(values.shape[0])
        features[position, :views] = values
        mask[position, :views] = True
    return ObservationBatch(
        features=features,
        mask=mask,
        labels=torch.tensor([int(item["label"]) for item in items], dtype=torch.long),
        observation_ids=torch.tensor(
            [int(item["observation_id"]) for item in items], dtype=torch.long
        ),
        budgets=torch.tensor([int(item["budget"]) for item in items], dtype=torch.long),
        photo_ids=[list(item["photo_ids"]) for item in items],
    )
