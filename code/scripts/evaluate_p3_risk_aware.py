from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from robird.budgets import stable_key
from robird.data import ObservationFeatureDataset, collate_observations, load_feature_table
from robird.evaluation import evaluate_loader, summarize_records
from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file
from robird.models import build_model
from robird.metrics import aggregate_subset_records, macro_top1
from robird.risk_aware import RiskAwareReliabilityAggregator
from robird.splits import validate_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen v5.3 P3 RLRA model.")
    parser.add_argument("--config", type=Path, default=Path("configs/p3_risk_aware_v5_3.yaml"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


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


def _class_count(manifest: pd.DataFrame) -> int:
    pairs = manifest[["taxon_id", "class_index"]].drop_duplicates()
    values = sorted(pairs["class_index"].astype(int).tolist())
    if pairs["taxon_id"].duplicated().any() or pairs["class_index"].duplicated().any() or values != list(range(len(values))):
        raise ValueError("Invalid taxon/class mapping")
    return len(values)


def _validate_inputs(config: Mapping[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    expected = config["expected"]
    resolved = {
        key: resolve_config_path(config, value)
        for key, value in paths.items()
        if key in {
            "manifest_csv", "splits_csv", "dataset_audit_json", "split_audit_json",
            "feature_matrix", "feature_index_csv", "feature_audit_json", "p1_checkpoint",
            "p1_gate_audit_json", "p2_result_json",
        }
    }
    pairs = {
        "manifest_csv": "manifest_sha256", "splits_csv": "splits_sha256",
        "dataset_audit_json": "dataset_audit_sha256", "split_audit_json": "split_audit_sha256",
        "feature_matrix": "feature_matrix_sha256", "feature_index_csv": "feature_index_sha256",
        "feature_audit_json": "feature_audit_sha256", "p1_checkpoint": "p1_checkpoint_sha256",
        "p1_gate_audit_json": "p1_gate_audit_sha256", "p2_result_json": "p2_result_sha256",
    }
    for path_key, hash_key in pairs.items():
        actual = sha256_file(resolved[path_key])
        if actual != str(expected[hash_key]):
            raise RuntimeError(f"Frozen {path_key} hash mismatch: {actual}")
    if _load_json(resolved["dataset_audit_json"]).get("decision") != "CONTINUE_DEVELOPMENT_ONLY":
        raise RuntimeError("P3 requires the v5.3 development-only Dataset P0 decision")
    if _load_json(resolved["split_audit_json"]).get("validation", {}).get("passed") is not True:
        raise RuntimeError("P3 requires a passing split audit")
    if _load_json(resolved["p1_gate_audit_json"]).get("status") != "STOP_P1_TASK_SIGNAL_GATE":
        raise RuntimeError("P3 requires the frozen P1 STOP evidence")
    if _load_json(resolved["p2_result_json"]).get("status") != "STOP_P2_HARMFUL_VIEW_GATE":
        raise RuntimeError("P3 requires the frozen P2 STOP evidence")
    return resolved


def _records(model: torch.nn.Module, manifest: pd.DataFrame, features: np.ndarray, feature_index: pd.DataFrame, budget: int, config: Mapping[str, Any], device: torch.device) -> list[dict[str, Any]]:
    data = ObservationFeatureDataset(
        manifest,
        features,
        feature_index,
        "development_test",
        int(budget),
        int(config["seed"]),
        "exhaustive",
        int(config["evaluation"]["max_subsets_per_budget"]),
    )
    loader = DataLoader(
        data,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_observations,
    )
    return evaluate_loader(model, loader, device)


def _pairwise_summary(records: list[dict[str, Any]], repeats: int, seed: int) -> dict[str, Any]:
    aggregated = aggregate_subset_records(records)
    first = {int(row["observation_id"]): row for row in aggregated if int(row["budget"]) == 1}
    second = {int(row["observation_id"]): row for row in aggregated if int(row["budget"]) == 2}
    ids = sorted(set(first) & set(second))
    if not ids:
        raise RuntimeError("No common budget-1-to-2 observations")
    labels = sorted({int(first[key]["label"]) for key in ids})
    by_species: list[np.ndarray] = []
    for label in labels:
        species_ids = [key for key in ids if int(first[key]["label"]) == label]
        values = np.asarray(
            [int(second[key]["prediction"] == label) - int(first[key]["prediction"] == label) for key in species_ids],
            dtype=np.float64,
        )
        by_species.append(values)
    point = float(np.mean([values.mean() for values in by_species]))
    rng = np.random.default_rng(int(seed))
    samples = np.zeros(int(repeats), dtype=np.float64)
    for values in by_species:
        samples += values[rng.integers(0, len(values), size=(int(repeats), len(values)))].mean(axis=1)
    samples /= len(by_species)
    corrections = sum(first[key]["prediction"] != first[key]["label"] and second[key]["prediction"] == second[key]["label"] for key in ids)
    regressions = sum(first[key]["prediction"] == first[key]["label"] and second[key]["prediction"] != second[key]["label"] for key in ids)
    def macro(rows: Mapping[int, dict[str, Any]]) -> float:
        return float(np.mean([np.mean([rows[key]["prediction"] == label for key in ids if rows[key]["label"] == label]) for label in labels]))
    return {
        "groups": len(ids),
        "species": len(labels),
        "budget1_macro_top1": macro(first),
        "budget2_macro_top1": macro(second),
        "delta": point,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "corrections": int(corrections),
        "regressions": int(regressions),
        "net_corrections": int(corrections - regressions),
    }


def _prediction_rows(method: str, records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    counters: defaultdict[tuple[int, int], int] = defaultdict(int)
    rows = []
    for record in records:
        key = (int(record["observation_id"]), int(record["budget"]))
        replicate = counters[key]
        counters[key] += 1
        photo_ids = [int(value) for value in record["photo_ids"]]
        weights = record.get("view_weights")
        rows.append({
            "method": method,
            "observation_id": key[0],
            "budget": key[1],
            "replicate": replicate,
            "subset_key": stable_key(int(seed), key[0], key[1], *photo_ids),
            "photo_ids": ";".join(str(value) for value in photo_ids),
            "label": int(record["label"]),
            "prediction": int(record["prediction"]),
            "probabilities_json": json.dumps(record["probabilities"].tolist(), separators=(",", ":")),
            "view_weights_json": json.dumps(weights.tolist() if hasattr(weights, "tolist") else [], separators=(",", ":")),
        })
    return rows


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    resolved = _validate_inputs(config)
    checkpoint_path = (args.checkpoint or resolve_config_path(config, config["paths"]["checkpoint"])).resolve()
    result_path = resolve_config_path(config, config["paths"]["result_json"])
    predictions_path = resolve_config_path(config, config["paths"]["predictions_csv"])
    if result_path.exists() or predictions_path.exists():
        raise FileExistsError("P3 evaluation artifacts already exist")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    provenance = checkpoint.get("provenance", {})
    expected_provenance = {
        "config_sha256": config["_config_hash"],
        "manifest_sha256": sha256_file(resolved["manifest_csv"]),
        "splits_sha256": sha256_file(resolved["splits_csv"]),
        "feature_index_sha256": sha256_file(resolved["feature_index_csv"]),
        "p1_checkpoint_sha256": sha256_file(resolved["p1_checkpoint"]),
        "p2_result_sha256": sha256_file(resolved["p2_result_json"]),
    }
    mismatches = {key: {"checkpoint": provenance.get(key), "current": value} for key, value in expected_provenance.items() if provenance.get(key) != value}
    if mismatches:
        raise RuntimeError(f"P3 checkpoint provenance mismatch: {mismatches}")
    manifest = _merge_splits(pd.read_csv(resolved["manifest_csv"]), pd.read_csv(resolved["splits_csv"]))
    features, feature_index = load_feature_table(resolved["feature_matrix"], resolved["feature_index_csv"])
    class_count = _class_count(manifest)
    model_config = provenance.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("P3 checkpoint lacks model_config")
    model = RiskAwareReliabilityAggregator(
        feature_dim=int(model_config["feature_dim"]),
        num_classes=class_count,
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        keep_probability=float(model_config["keep_probability"]),
        anchor_probability=float(model_config["anchor_probability"]),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    baseline_checkpoint = torch.load(resolved["p1_checkpoint"], map_location="cpu")
    baseline_config = baseline_checkpoint.get("provenance", {}).get("model_config")
    baseline = build_model(baseline_config, num_classes=class_count)
    baseline.load_state_dict(baseline_checkpoint["model_state"], strict=True)
    device = _device(args.device)
    model.to(device).eval()
    baseline.to(device).eval()
    model_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    for budget in [int(value) for value in config["evaluation"]["budgets"]]:
        model_records.extend(_records(model, manifest, features, feature_index, budget, config, device))
        baseline_records.extend(_records(baseline, manifest, features, feature_index, budget, config, device))
        print(json.dumps({"budget": budget, "model_records": len(model_records), "baseline_records": len(baseline_records)}), flush=True)
    model_metrics = summarize_records(model_records, int(config["evaluation"]["ece_bins"]), topk=[1, 5])
    baseline_metrics = summarize_records(baseline_records, int(config["evaluation"]["ece_bins"]), topk=[1, 5])
    repeats = int(config["evaluation"]["bootstrap_repeats"])
    model_pair = _pairwise_summary(model_records, repeats, int(config["seed"]))
    baseline_pair = _pairwise_summary(baseline_records, repeats, int(config["seed"]) + 1)
    expected_groups = int(config["evaluation"]["expected_two_view_groups"])
    if model_pair["groups"] != expected_groups or baseline_pair["groups"] != expected_groups:
        raise RuntimeError(f"Unexpected budget-1-to-2 cohort: model={model_pair['groups']} baseline={baseline_pair['groups']}")
    gates = {
        "recovery_vs_mean_logit": model_pair["budget2_macro_top1"] - baseline_pair["budget2_macro_top1"] >= float(config["gate"]["min_budget2_recovery_vs_mean"]),
        "delta_not_worse_than_minus_0_3pp": model_pair["delta"] >= float(config["gate"]["min_budget2_minus_budget1_delta"]),
        "positive_net_corrections": model_pair["net_corrections"] > 0,
    }
    prediction_rows = _prediction_rows("rlra", model_records, int(config["seed"])) + _prediction_rows("mean_logit", baseline_records, int(config["seed"]))
    atomic_write_csv(predictions_path, pd.DataFrame.from_records(prediction_rows), refuse_if_exists=True)
    result = {
        "status": "PASS_P3_RISK_AWARE_GATE" if all(gates.values()) else "STOP_P3_RISK_AWARE_GATE",
        "gate_pass": bool(all(gates.values())),
        "gates": gates,
        "primary_method": "rlra",
        "by_method": {"rlra": model_metrics, "mean_logit": baseline_metrics},
        "common_budget12": {"rlra": model_pair, "mean_logit": baseline_pair},
        "provenance": {
            "config_sha256": config["_config_hash"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "manifest_sha256": sha256_file(resolved["manifest_csv"]),
            "splits_sha256": sha256_file(resolved["splits_csv"]),
            "feature_matrix_sha256": sha256_file(resolved["feature_matrix"]),
            "feature_index_sha256": sha256_file(resolved["feature_index_csv"]),
            "p1_checkpoint_sha256": sha256_file(resolved["p1_checkpoint"]),
            "p2_result_sha256": sha256_file(resolved["p2_result_json"]),
            "predictions_sha256": sha256_file(predictions_path),
            "claim_scope": "CONTINUE_DEVELOPMENT_ONLY",
        },
    }
    atomic_write_json(result_path, result, refuse_if_exists=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
