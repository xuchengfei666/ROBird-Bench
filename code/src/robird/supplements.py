from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from robird.data import ObservationFeatureDataset, collate_observations, load_feature_table
from robird.evaluation import evaluate_loader, summarize_records
from robird.io import resolve_config_path, sha256_file
from robird.models import build_model
from robird.splits import validate_splits


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def merge_splits(manifest: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    required = {"observation_id", "observer_id", "taxon_id", "split"}
    missing = sorted(required - set(splits.columns))
    if missing or splits["observation_id"].duplicated().any():
        raise ValueError(f"Invalid split table; missing={missing}")
    joined = manifest.drop(columns=["split"], errors="ignore").merge(
        splits[["observation_id", "observer_id", "taxon_id", "split"]],
        on="observation_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_split"),
    )
    if joined["split"].isna().any():
        raise ValueError("Some manifest observations have no split")
    for column in ("observer_id", "taxon_id"):
        other = f"{column}_split"
        if not joined[column].eq(joined[other]).all():
            raise ValueError(f"Manifest/split mismatch in {column}")
        joined = joined.drop(columns=other)
    validation = validate_splits(joined)
    if not validation["passed"]:
        raise RuntimeError(f"Split leakage validation failed: {validation['violations']}")
    return joined


def load_context(config: Mapping[str, Any]) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    paths = config["paths"]
    audit_path = resolve_config_path(config, paths["dataset_audit_json"])
    manifest_path = resolve_config_path(config, paths["manifest_csv"])
    audit = load_json(audit_path)
    if audit.get("decision") not in {"CONTINUE_DEVELOPMENT_ONLY", "CONTINUE_TO_COMMON_BENCHMARK"}:
        raise RuntimeError(f"Dataset P0 has not passed: {audit.get('decision')}")
    if audit.get("provenance", {}).get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("Dataset audit does not belong to the current manifest")
    split_path = resolve_config_path(config, paths["splits_csv"])
    manifest = merge_splits(pd.read_csv(manifest_path), pd.read_csv(split_path))
    matrix_path = resolve_config_path(config, paths["feature_matrix"])
    index_path = resolve_config_path(config, paths["feature_index_csv"])
    features, feature_index = load_feature_table(matrix_path, index_path)
    return manifest, features, feature_index


def make_loader(dataset: ObservationFeatureDataset, batch_size: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_observations,
    )


def load_checkpoint_model(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"Checkpoint lacks model_state: {checkpoint_path}")
    provenance = checkpoint.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Checkpoint lacks provenance: {checkpoint_path}")
    model_config = provenance.get("model_config")
    class_map = provenance.get("class_map")
    if not isinstance(model_config, dict) or not isinstance(class_map, dict):
        raise ValueError(f"Checkpoint lacks model_config/class_map: {checkpoint_path}")
    model = build_model(model_config, num_classes=len(class_map))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, provenance


def evaluate_manifest(
    model: torch.nn.Module,
    manifest: pd.DataFrame,
    features: np.ndarray,
    feature_index: pd.DataFrame,
    config: Mapping[str, Any],
    split: str,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    data_config = config["data"]
    eval_config = config["evaluation"]
    for budget in [int(value) for value in eval_config["budgets"]]:
        dataset = ObservationFeatureDataset(
            manifest,
            features,
            feature_index,
            split=split,
            budget=budget,
            seed=int(config["seed"]),
            subset_mode=str(data_config["eval_subset_mode"]),
            max_subsets=int(data_config["max_subsets_per_budget"]),
        )
        if len(dataset):
            records.extend(
                evaluate_loader(
                    model,
                    make_loader(dataset, int(config["training"]["batch_size"]), device),
                    device,
                )
            )
    if not records:
        raise RuntimeError(f"No evaluable observations for split {split!r}")
    return summarize_records(
        records,
        ece_bins=int(eval_config["ece_bins"]),
        topk=[int(value) for value in eval_config["topk"]],
    ), records


def group_photo_map(manifest: pd.DataFrame, split: str) -> dict[int, list[int]]:
    selected = manifest.loc[manifest["split"] == split]
    output: dict[int, list[int]] = {}
    for observation_id, group in selected.groupby("observation_id", sort=True):
        output[int(observation_id)] = sorted(int(value) for value in group["photo_id"])
    return output


def build_circular_shuffle_manifest(manifest: pd.DataFrame, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected = manifest.loc[manifest["split"] == split].copy()
    selected["photo_count"] = selected.groupby("observation_id")["photo_id"].transform("count")
    eligible_ids: list[int] = []
    source_for_destination: dict[int, int] = {}
    excluded_singletons = 0
    stratum_counts: dict[str, int] = {}
    for (taxon_id, photo_count), group in selected.groupby(["taxon_id", "photo_count"], sort=True):
        observation_ids = sorted(int(value) for value in group["observation_id"].unique())
        key = f"taxon={int(taxon_id)}|photos={int(photo_count)}"
        stratum_counts[key] = len(observation_ids)
        if len(observation_ids) < 2:
            excluded_singletons += len(observation_ids)
            continue
        eligible_ids.extend(observation_ids)
        for index, destination in enumerate(observation_ids):
            source_for_destination[destination] = observation_ids[(index + 1) % len(observation_ids)]

    source_rows = selected.set_index("observation_id")
    rows: list[dict[str, Any]] = []
    for destination in sorted(eligible_ids):
        source = source_for_destination[destination]
        destination_rows = selected.loc[selected["observation_id"] == destination]
        source_rows_for_group = selected.loc[selected["observation_id"] == source].sort_values("photo_id")
        if len(destination_rows) != len(source_rows_for_group):
            raise RuntimeError("Circular shuffle changed group cardinality")
        base = destination_rows.iloc[0].to_dict()
        for source_row in source_rows_for_group.itertuples(index=False):
            row = dict(base)
            row["photo_id"] = int(source_row.photo_id)
            row["local_path"] = source_row.local_path
            row["sha256"] = source_row.sha256
            row["width"] = int(source_row.width)
            row["height"] = int(source_row.height)
            row["url"] = source_row.url
            row["attribution"] = source_row.attribution
            row["license_code"] = source_row.license_code
            rows.append(row)
    shuffled = pd.DataFrame.from_records(rows).drop(columns=["photo_count"], errors="ignore")
    if shuffled.empty:
        raise RuntimeError("No eligible same-taxon strata for shuffled control")
    metadata = {
        "split": split,
        "eligible_observations": len(eligible_ids),
        "excluded_singleton_observations": excluded_singletons,
        "strata": stratum_counts,
        "assignment": {str(destination): int(source) for destination, source in sorted(source_for_destination.items())},
    }
    return shuffled, metadata


def correctness_by_observation(records: list[dict[str, Any]]) -> dict[tuple[int, int], bool]:
    grouped: dict[tuple[int, int], list[bool]] = defaultdict(list)
    for record in records:
        grouped[(int(record["observation_id"]), int(record["budget"]))].append(
            int(record["prediction"]) == int(record["label"])
        )
    return {key: bool(np.mean(values) >= 0.5) for key, values in grouped.items()}
