from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import DataLoader

from robird.data import ObservationFeatureDataset, collate_observations, load_feature_table
from robird.io import (
    atomic_write_json,
    load_yaml,
    require_frozen_protocol,
    resolve_config_path,
    sha256_file,
)
from robird.models import build_model
from robird.splits import validate_splits
from robird.training import fit, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one ROBird feature-level aggregator.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--model", type=str)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _require_dataset_pass(audit_path: Path, manifest_path: Path) -> None:
    audit = _load_json(audit_path)
    allowed = {"CONTINUE_DEVELOPMENT_ONLY", "CONTINUE_TO_COMMON_BENCHMARK"}
    if audit.get("decision") not in allowed:
        raise RuntimeError(f"Dataset P0 has not passed: {audit.get('decision', 'missing decision')}")
    audited_manifest_hash = audit.get("provenance", {}).get("manifest_sha256")
    current_manifest_hash = sha256_file(manifest_path)
    if audited_manifest_hash != current_manifest_hash:
        raise RuntimeError(
            "Dataset audit does not belong to the current manifest: "
            f"audit={audited_manifest_hash}, current={current_manifest_hash}"
        )


def _merge_splits(manifest: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    required = {"observation_id", "observer_id", "taxon_id", "split"}
    missing = sorted(required - set(splits.columns))
    if missing:
        raise ValueError(f"Split table missing columns: {missing}")
    if splits["observation_id"].duplicated().any():
        raise ValueError("Split table contains duplicate observation_id rows")
    base = manifest.drop(columns=["split"], errors="ignore")
    joined = base.merge(
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


def _class_map(manifest: pd.DataFrame) -> dict[int, int]:
    pairs = manifest[["taxon_id", "class_index"]].drop_duplicates()
    if pairs["taxon_id"].duplicated().any() or pairs["class_index"].duplicated().any():
        raise ValueError("Taxon/class mapping is not one-to-one")
    mapping = {int(row.taxon_id): int(row.class_index) for row in pairs.itertuples(index=False)}
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("class_index values must be contiguous from zero")
    return mapping


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _loader(dataset: ObservationFeatureDataset, config: Mapping[str, Any], shuffle: bool, device: torch.device) -> DataLoader:
    workers = int(config["data"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_observations,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    audit_path = resolve_config_path(config, paths["dataset_audit_json"])
    manifest_path = resolve_config_path(config, paths["manifest_csv"])
    _require_dataset_pass(audit_path, manifest_path)
    splits_path = resolve_config_path(config, paths["splits_csv"])
    matrix_path = resolve_config_path(config, paths["feature_matrix"])
    index_path = resolve_config_path(config, paths["feature_index_csv"])
    manifest = _merge_splits(pd.read_csv(manifest_path), pd.read_csv(splits_path))
    class_map = _class_map(manifest)
    features, feature_index = load_feature_table(matrix_path, index_path)

    model_config = dict(config["model"])
    if args.model:
        model_config["name"] = args.model
    model_name = str(model_config["name"]).lower()
    budget_value = args.budget if args.budget is not None else config["training"].get("budget")
    budget = None if budget_value is None else int(budget_value)
    if budget is not None and budget < 1:
        raise ValueError("Training budget must be positive")
    run_name = args.run_name or f"{model_name}-{'all' if budget is None else f'k{budget}'}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("run-name may contain only letters, digits, dot, underscore and hyphen")
    result_dir = resolve_config_path(config, paths["result_root"]) / run_name
    checkpoint_path = result_dir / "best.pth"
    history_path = result_dir / "history.json"
    if not args.overwrite:
        existing = [str(path) for path in (checkpoint_path, history_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite training artifacts: {existing}")

    seed = int(config["seed"])
    seed_everything(seed)
    device = _device(args.device)
    data_config = config["data"]
    train_data = ObservationFeatureDataset(
        manifest,
        features,
        feature_index,
        split="train",
        budget=budget,
        seed=seed,
        subset_mode=str(data_config["train_subset_mode"]),
        max_subsets=int(data_config["max_subsets_per_budget"]),
    )
    validation_data = ObservationFeatureDataset(
        manifest,
        features,
        feature_index,
        split="validation",
        budget=budget,
        seed=seed,
        subset_mode=str(data_config["eval_subset_mode"]),
        max_subsets=int(data_config["max_subsets_per_budget"]),
    )
    if not len(train_data) or not len(validation_data):
        raise RuntimeError("Train or validation dataset is empty")

    model = build_model(model_config, num_classes=len(class_map))
    provenance = {
        "protocol": dict(config["protocol"]),
        "config_sha256": config["_config_hash"],
        "dataset_audit_sha256": sha256_file(audit_path),
        "manifest_sha256": sha256_file(manifest_path),
        "splits_sha256": sha256_file(splits_path),
        "feature_index_sha256": sha256_file(index_path),
        "class_map": class_map,
        "model_config": model_config,
        "training_budget": budget,
        "run_name": run_name,
        "seed": seed,
    }
    history = fit(
        model,
        _loader(train_data, config, shuffle=True, device=device),
        _loader(validation_data, config, shuffle=False, device=device),
        config,
        device,
        checkpoint_path,
        provenance,
    )
    atomic_write_json(history_path, history, refuse_if_exists=not args.overwrite)
    print(json.dumps({"checkpoint": str(checkpoint_path), "epochs": len(history)}, indent=2))


if __name__ == "__main__":
    main()
