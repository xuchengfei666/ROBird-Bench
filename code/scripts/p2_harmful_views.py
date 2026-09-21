from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import laplace

from robird.budgets import enumerate_budget_subsets
from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file
from robird.models import build_model
from robird.metrics import macro_top1, negative_log_likelihood, brier_score, expected_calibration_error


METHODS = ["mean_logit", "median_logit", "quality_top_half", "consensus_top_half", "quality_consensus_trim"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen v5.3 harmful-view mechanism pilot.")
    parser.add_argument("--config", type=Path, default=Path("configs/p2_harmful_views_v5_3.yaml"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) <= 1 or float(values.std()) < 1e-12:
        return np.zeros(len(values), dtype=np.float64)
    return (values - values.mean()) / values.std()


def quality_score(path: Path, resize: int) -> tuple[float, float, float]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((resize, resize), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    focus = float(np.var(laplace(gray)))
    contrast = float(np.std(gray))
    exposure = float(1.0 - 2.0 * abs(float(gray.mean()) - 0.5))
    return float(np.log1p(focus)), contrast, exposure


def select_indices(
    method: str,
    logits: np.ndarray,
    quality: np.ndarray,
    photo_ids: list[int],
    keep_fraction: float = 0.5,
) -> list[int]:
    views = len(photo_ids)
    if not 0.0 < float(keep_fraction) <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    keep = max(1, int(np.ceil(views * float(keep_fraction))))
    if views == 1:
        return [0]
    centered = logits - logits.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = centered / np.clip(norms, 1e-12, None)
    consensus = normalized.mean(axis=0)
    consensus /= max(float(np.linalg.norm(consensus)), 1e-12)
    consensus_score = normalized @ consensus
    if method == "mean_logit":
        return list(range(views))
    if method == "median_logit":
        return list(range(views))
    if method == "quality_top_half":
        scores = np.asarray(quality, dtype=np.float64)
    elif method == "consensus_top_half":
        scores = consensus_score
    elif method == "quality_consensus_trim":
        scores = 0.5 * np.asarray(quality, dtype=np.float64) + 0.5 * consensus_score
    else:
        raise ValueError(f"Unknown method: {method}")
    return sorted(range(views), key=lambda index: (-float(scores[index]), int(photo_ids[index])))[:keep]


def aggregate(
    method: str,
    logits: np.ndarray,
    quality: np.ndarray,
    photo_ids: list[int],
    keep_fraction: float = 0.5,
) -> tuple[np.ndarray, list[int]]:
    selected = select_indices(method, logits, quality, photo_ids, keep_fraction)
    if method == "median_logit":
        return np.median(logits, axis=0), selected
    return np.mean(logits[selected], axis=0), selected


def summarize(rows: list[dict[str, Any]], bootstrap_repeats: int, seed: int) -> dict[str, Any]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), int(row["budget"]), int(row["observation_id"])), []).append(row)
    agg: list[dict[str, Any]] = []
    for (method, budget, observation_id), values in sorted(grouped.items()):
        probabilities = np.mean(np.stack([v["probabilities"] for v in values]), axis=0)
        label = int(values[0]["label"])
        agg.append({"method": method, "budget": budget, "observation_id": observation_id, "label": label, "probabilities": probabilities, "prediction": int(np.argmax(probabilities))})
    methods = sorted({str(row["method"]) for row in agg})
    budgets = sorted({int(row["budget"]) for row in agg})
    results: dict[str, Any] = {"by_method": {}, "common_budget12": {}}
    rng = np.random.default_rng(seed)
    for method in methods:
        results["by_method"][method] = {}
        for budget in budgets:
            values = [r for r in agg if r["method"] == method and r["budget"] == budget]
            probs = np.stack([r["probabilities"] for r in values])
            labels = np.asarray([r["label"] for r in values], dtype=np.int64)
            results["by_method"][method][str(budget)] = {"groups": len(values), "macro_top1": macro_top1(probs, labels), "micro_top1": float(np.mean(np.argmax(probs, axis=1) == labels)), "nll": negative_log_likelihood(probs, labels), "brier": brier_score(probs, labels), "ece": expected_calibration_error(probs, labels)}
        first = {r["observation_id"]: r for r in agg if r["method"] == method and r["budget"] == 1}
        second = {r["observation_id"]: r for r in agg if r["method"] == method and r["budget"] == 2}
        ids = sorted(set(first) & set(second))
        labels = sorted({first[i]["label"] for i in ids})
        per_species = []
        for label in labels:
            species_ids = [i for i in ids if first[i]["label"] == label]
            delta_values = np.asarray([int(second[i]["prediction"] == label) - int(first[i]["prediction"] == label) for i in species_ids], dtype=np.float64)
            per_species.append(delta_values)
        point = float(np.mean([v.mean() for v in per_species]))
        samples = np.zeros(int(bootstrap_repeats), dtype=np.float64)
        for values in per_species:
            samples += values[rng.integers(0, len(values), size=(int(bootstrap_repeats), len(values)))].mean(axis=1)
        samples /= len(per_species)
        corrections = sum(first[i]["prediction"] != first[i]["label"] and second[i]["prediction"] == second[i]["label"] for i in ids)
        regressions = sum(first[i]["prediction"] == first[i]["label"] and second[i]["prediction"] != second[i]["label"] for i in ids)
        results["common_budget12"][method] = {"groups": len(ids), "species": len(labels), "budget1_macro_top1": float(np.mean([np.mean([first[i]["prediction"] == label for i in ids if first[i]["label"] == label]) for label in labels])), "budget2_macro_top1": float(np.mean([np.mean([second[i]["prediction"] == label for i in ids if first[i]["label"] == label]) for label in labels])), "delta": point, "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))], "corrections": int(corrections), "regressions": int(regressions), "net_corrections": int(corrections - regressions)}
    return results


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    expected = config["expected"]
    manifest_path = resolve_config_path(config, paths["manifest_csv"])
    splits_path = resolve_config_path(config, paths["splits_csv"])
    audit_path = resolve_config_path(config, paths["dataset_audit_json"])
    split_audit_path = resolve_config_path(config, paths["split_audit_json"])
    p1_gate_path = resolve_config_path(config, paths["p1_gate_audit_json"])
    feature_path = resolve_config_path(config, paths["feature_matrix"])
    feature_index_path = resolve_config_path(config, paths["feature_index_csv"])
    checkpoint_path = resolve_config_path(config, paths["checkpoint"])
    result_path = resolve_config_path(config, paths["result_json"])
    predictions_path = resolve_config_path(config, paths["predictions_csv"])
    quality_path = resolve_config_path(config, paths["quality_cache_csv"])
    if result_path.exists() or predictions_path.exists() or quality_path.exists():
        raise FileExistsError("P2 outputs already exist")
    for path, value, label in ((manifest_path, expected["manifest_sha256"], "manifest"), (splits_path, expected["splits_sha256"], "splits"), (audit_path, expected["dataset_audit_sha256"], "audit"), (split_audit_path, expected["split_audit_sha256"], "split_audit"), (feature_path, expected["feature_matrix_sha256"], "feature_matrix"), (feature_index_path, expected["feature_index_sha256"], "feature_index"), (checkpoint_path, expected["checkpoint_sha256"], "checkpoint")):
        actual = sha256_file(path)
        if actual != str(value):
            raise RuntimeError(f"Frozen {label} hash mismatch: {actual}")
    audit = load_json(audit_path)
    split_audit = load_json(split_audit_path)
    p1_gate = load_json(p1_gate_path)
    if audit.get("decision") != "CONTINUE_DEVELOPMENT_ONLY" or split_audit.get("validation", {}).get("passed") is not True:
        raise RuntimeError("P2 requires a passing development-only Dataset P0 and split audit")
    if p1_gate.get("status") != "STOP_P1_TASK_SIGNAL_GATE" or p1_gate.get("gate_pass") is not False:
        raise RuntimeError("P2 requires the frozen negative P1 task-signal gate")
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    splits = pd.read_csv(splits_path)
    manifest = manifest.merge(splits[["observation_id", "split"]], on="observation_id", how="left", validate="many_to_one")
    features = np.load(feature_path, mmap_mode="r")
    feature_index = pd.read_csv(feature_index_path)
    feature_rows = dict(zip(feature_index.photo_id.astype(int), feature_index.feature_row.astype(int), strict=True))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    provenance = checkpoint.get("provenance", {})
    if provenance.get("manifest_sha256") != str(expected["manifest_sha256"]) or provenance.get("feature_index_sha256") != str(expected["feature_index_sha256"]):
        raise RuntimeError("P1 checkpoint provenance does not match P2 inputs")
    model_config = provenance.get("model_config")
    if not isinstance(model_config, dict) or str(model_config.get("name", "")).lower() != "mean_feature":
        raise RuntimeError("P2 requires the frozen mean-feature-k1 checkpoint")
    model = build_model(model_config, num_classes=manifest.class_index.nunique())
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    weight = model.classifier.weight.detach().numpy().astype(np.float64)
    bias = model.classifier.bias.detach().numpy().astype(np.float64)
    quality_rows = []
    quality_values: dict[int, tuple[float, float, float]] = {}
    for row in manifest.sort_values("row_id").itertuples(index=False):
        values = quality_score(Path(str(row.local_path)), int(config["quality"]["resize"]))
        quality_values[int(row.photo_id)] = values
        quality_rows.append({"photo_id": int(row.photo_id), "observation_id": int(row.observation_id), "log_laplacian_variance": values[0], "contrast": values[1], "exposure": values[2]})
    atomic_write_csv(quality_path, pd.DataFrame.from_records(quality_rows), refuse_if_exists=True)
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for observation_id, group in manifest[manifest["split"] == "development_test"].groupby("observation_id", sort=True):
        photo_ids = sorted(group.photo_id.astype(int).tolist())
        label = int(group.class_index.iloc[0])
        for budget in [int(v) for v in config["evaluation"]["budgets"]]:
            if budget > len(photo_ids):
                continue
            for subset_index, subset in enumerate(enumerate_budget_subsets(photo_ids, budget, 32, int(config["seed"]), int(observation_id))):
                values = np.asarray(features[[feature_rows[p] for p in subset]], dtype=np.float64)
                logits = values @ weight.T + bias
                components = np.asarray([quality_values[p] for p in subset], dtype=np.float64)
                quality = (
                    float(config["quality"]["laplacian_weight"]) * zscore(components[:, 0])
                    + float(config["quality"]["contrast_weight"]) * zscore(components[:, 1])
                    + float(config["quality"]["exposure_weight"]) * zscore(components[:, 2])
                )
                for method in METHODS:
                    pooled, selected = aggregate(
                        method,
                        logits,
                        quality,
                        list(subset),
                        float(config["quality"]["keep_fraction"]),
                    )
                    shifted = pooled - np.max(pooled)
                    probabilities = np.exp(shifted) / np.exp(shifted).sum()
                    rows.append({"method": method, "observation_id": int(observation_id), "budget": budget, "label": label, "probabilities": probabilities})
                    prediction_rows.append({"method": method, "observation_id": int(observation_id), "budget": budget, "replicate": subset_index, "photo_ids": ";".join(map(str, subset)), "selected_photo_ids": ";".join(str(subset[i]) for i in selected), "label": label, "prediction": int(np.argmax(probabilities)), "probabilities_json": json.dumps(probabilities.tolist(), separators=(",", ":"))})
    result = summarize(rows, int(config["evaluation"]["bootstrap_repeats"]), int(config["seed"]))
    primary = result["common_budget12"]["quality_consensus_trim"]
    mean_baseline = result["common_budget12"]["mean_logit"]
    expected_two_view_groups = int(config["expected"]["expected_two_view_groups"])
    if int(primary["groups"]) != expected_two_view_groups:
        raise RuntimeError(
            "Unexpected budget-1-to-2 cohort size: "
            f"expected {expected_two_view_groups}, got {primary['groups']}"
        )
    gates = {"recovery_vs_mean_logit": primary["budget2_macro_top1"] - mean_baseline["budget2_macro_top1"] >= 0.005, "delta_not_worse_than_minus_0_3pp": primary["delta"] >= -0.003, "positive_net_corrections": primary["net_corrections"] > 0}
    result["status"] = "PASS_P2_HARMFUL_VIEW_GATE" if all(gates.values()) else "STOP_P2_HARMFUL_VIEW_GATE"
    result["gate_pass"] = bool(all(gates.values()))
    result["gates"] = gates
    result["primary_method"] = "quality_consensus_trim"
    result["provenance"] = {"config_sha256": config["_config_hash"], "manifest_sha256": sha256_file(manifest_path), "splits_sha256": sha256_file(splits_path), "feature_matrix_sha256": sha256_file(feature_path), "feature_index_sha256": sha256_file(feature_index_path), "checkpoint_sha256": sha256_file(checkpoint_path), "quality_cache_sha256": sha256_file(quality_path)}
    atomic_write_csv(predictions_path, pd.DataFrame.from_records(prediction_rows), refuse_if_exists=True)
    atomic_write_json(result_path, result, refuse_if_exists=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
