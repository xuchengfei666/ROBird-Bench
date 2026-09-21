from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from robird.budgets import stable_key
from robird.data import ObservationFeatureDataset, collate_observations, load_feature_table
from robird.evaluation import evaluate_loader, summarize_records
from robird.io import (
    atomic_write_csv,
    atomic_write_json,
    load_yaml,
    require_frozen_protocol,
    resolve_config_path,
    sha256_file,
)
from robird.models import build_model
from robird.metrics import aggregate_subset_records, macro_top1, stratified_group_bootstrap
from robird.splits import validate_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ROBird checkpoint by photo budget.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="development_test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _merge_splits(manifest: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
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


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _check_provenance(provenance: dict[str, Any], expected: dict[str, str]) -> None:
    mismatches = {
        key: {"checkpoint": provenance.get(key), "current": value}
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint provenance mismatch: {mismatches}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    audit_path = resolve_config_path(config, paths["dataset_audit_json"])
    manifest_path = resolve_config_path(config, paths["manifest_csv"])
    audit = _load_json(audit_path)
    allowed = {"CONTINUE_DEVELOPMENT_ONLY", "CONTINUE_TO_COMMON_BENCHMARK"}
    if audit.get("decision") not in allowed:
        raise RuntimeError(f"Dataset P0 has not passed: {audit.get('decision', 'missing decision')}")
    if args.split != "development_test" and audit.get("decision") != "CONTINUE_TO_COMMON_BENCHMARK":
        raise RuntimeError("Only development_test may be evaluated before the unseen holdout is registered")
    audited_manifest_hash = audit.get("provenance", {}).get("manifest_sha256")
    current_manifest_hash = sha256_file(manifest_path)
    if audited_manifest_hash != current_manifest_hash:
        raise RuntimeError(
            "Dataset audit does not belong to the current manifest: "
            f"audit={audited_manifest_hash}, current={current_manifest_hash}"
        )

    splits_path = resolve_config_path(config, paths["splits_csv"])
    matrix_path = resolve_config_path(config, paths["feature_matrix"])
    index_path = resolve_config_path(config, paths["feature_index_csv"])
    manifest = _merge_splits(pd.read_csv(manifest_path), pd.read_csv(splits_path))
    features, feature_index = load_feature_table(matrix_path, index_path)

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint does not contain a model_state mapping")
    provenance = checkpoint.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("Checkpoint provenance is missing")
    _check_provenance(
        provenance,
        {
            "config_sha256": config["_config_hash"],
            "dataset_audit_sha256": sha256_file(audit_path),
            "manifest_sha256": sha256_file(manifest_path),
            "splits_sha256": sha256_file(splits_path),
            "feature_index_sha256": sha256_file(index_path),
        },
    )
    model_config = provenance.get("model_config")
    class_map = provenance.get("class_map")
    if not isinstance(model_config, dict) or not isinstance(class_map, dict):
        raise ValueError("Checkpoint lacks model_config or class_map")
    model = build_model(model_config, num_classes=len(class_map))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = _device(args.device)
    output_dir = args.output_dir.resolve() if args.output_dir else checkpoint_path.parent
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"
    if not args.overwrite:
        existing = [str(path) for path in (metrics_path, predictions_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite evaluation artifacts: {existing}")

    records: list[dict[str, Any]] = []
    data_config = config["data"]
    workers = int(data_config["num_workers"])
    for budget in [int(value) for value in config["evaluation"]["budgets"]]:
        dataset = ObservationFeatureDataset(
            manifest,
            features,
            feature_index,
            split=args.split,
            budget=budget,
            seed=int(config["seed"]),
            subset_mode=str(data_config["eval_subset_mode"]),
            max_subsets=int(data_config["max_subsets_per_budget"]),
        )
        if not len(dataset):
            continue
        loader = DataLoader(
            dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
            persistent_workers=workers > 0,
            collate_fn=collate_observations,
        )
        records.extend(evaluate_loader(model, loader, device))
    if not records:
        raise RuntimeError(f"No evaluable observations for split {args.split!r}")

    evaluation_config = config["evaluation"]
    metrics = summarize_records(
        records,
        ece_bins=int(evaluation_config["ece_bins"]),
        topk=[int(value) for value in evaluation_config["topk"]],
    )
    aggregated = aggregate_subset_records(records)
    budgets = sorted({int(row["budget"]) for row in aggregated})
    common_ids = set.intersection(
        *[
            {int(row["observation_id"]) for row in aggregated if int(row["budget"]) == budget}
            for budget in budgets
        ]
    )
    common_rows = [row for row in aggregated if int(row["observation_id"]) in common_ids]
    repeats = int(evaluation_config["bootstrap_repeats"])

    def macro_metric(rows: list[dict[str, Any]]) -> float:
        probabilities = np.stack([row["probabilities"] for row in rows])
        labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        return macro_top1(probabilities, labels)

    per_budget_bootstrap = {}
    for budget in budgets:
        rows = [row for row in common_rows if int(row["budget"]) == budget]
        per_budget_bootstrap[str(budget)] = stratified_group_bootstrap(
            rows, macro_metric, repeats, int(config["seed"]) + budget
        )

    by_observation: dict[int, dict[str, Any]] = {}
    for row in common_rows:
        observation_id = int(row["observation_id"])
        trajectory = by_observation.setdefault(
            observation_id,
            {"observation_id": observation_id, "label": int(row["label"]), "by_budget": {}},
        )
        trajectory["by_budget"][int(row["budget"])] = row["probabilities"]

    def aubc_metric(rows: list[dict[str, Any]]) -> float:
        values = []
        for budget in budgets:
            probabilities = np.stack([row["by_budget"][budget] for row in rows])
            labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
            values.append(macro_top1(probabilities, labels))
        if len(budgets) == 1:
            return values[0]
        return float(np.trapezoid(values, budgets) / (budgets[-1] - budgets[0]))

    metrics["bootstrap"] = {
        "common_cohort_macro_top1_by_budget": per_budget_bootstrap,
        "common_cohort_normalized_macro_aubc": stratified_group_bootstrap(
            list(by_observation.values()), aubc_metric, repeats, int(config["seed"])
        ),
    }
    metrics["provenance"] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "split": args.split,
        "subset_unit": "observation_budget_subset",
        "claim_scope": audit["decision"],
        "bootstrap_repeats": repeats,
    }
    counters: defaultdict[tuple[int, int], int] = defaultdict(int)
    prediction_rows: list[dict[str, Any]] = []
    for record in records:
        key = (int(record["observation_id"]), int(record["budget"]))
        replicate = counters[key]
        counters[key] += 1
        photo_ids = [int(value) for value in record["photo_ids"]]
        prediction_rows.append(
            {
                "observation_id": key[0],
                "budget": key[1],
                "replicate": replicate,
                "subset_key": stable_key(int(config["seed"]), key[0], key[1], *photo_ids),
                "photo_ids": ";".join(str(value) for value in photo_ids),
                "label": int(record["label"]),
                "prediction": int(record["prediction"]),
                "probabilities_json": json.dumps(record["probabilities"].tolist(), separators=(",", ":")),
                "view_weights_json": json.dumps(
                    record.get("view_weights", []).tolist()
                    if hasattr(record.get("view_weights", []), "tolist")
                    else record.get("view_weights", []),
                    separators=(",", ":"),
                ),
            }
        )

    atomic_write_csv(predictions_path, pd.DataFrame.from_records(prediction_rows), refuse_if_exists=not args.overwrite)
    atomic_write_json(metrics_path, metrics, refuse_if_exists=not args.overwrite)
    print(json.dumps({"metrics": str(metrics_path), "predictions": str(predictions_path)}, indent=2))


if __name__ == "__main__":
    main()
