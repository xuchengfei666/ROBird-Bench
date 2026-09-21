from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file
from robird.metrics import aggregate_subset_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the frozen v5.3 P4 comparative suite.")
    parser.add_argument("--config", type=Path, default=Path("configs/p4_full_benchmark_v5_3.yaml"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _records(path: Path, method: str | None = None) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    if method is not None:
        if "method" in frame.columns:
            frame = frame.loc[frame["method"] == method].copy()
    records = []
    for row in frame.itertuples(index=False):
        probabilities = np.asarray(json.loads(row.probabilities_json), dtype=np.float64)
        records.append({
            "observation_id": int(row.observation_id),
            "budget": int(row.budget),
            "label": int(row.label),
            "probabilities": probabilities,
            "prediction": int(row.prediction),
        })
    return records


def _pairwise(records: list[dict[str, Any]], repeats: int, seed: int) -> dict[str, Any]:
    aggregate = aggregate_subset_records(records)
    first = {int(row["observation_id"]): row for row in aggregate if int(row["budget"]) == 1}
    second = {int(row["observation_id"]): row for row in aggregate if int(row["budget"]) == 2}
    ids = sorted(set(first) & set(second))
    labels = sorted({int(first[key]["label"]) for key in ids})
    values_by_species = []
    for label in labels:
        species_ids = [key for key in ids if int(first[key]["label"]) == label]
        values_by_species.append(np.asarray([int(second[key]["prediction"] == label) - int(first[key]["prediction"] == label) for key in species_ids], dtype=np.float64))
    point = float(np.mean([values.mean() for values in values_by_species]))
    rng = np.random.default_rng(int(seed))
    samples = np.zeros(int(repeats), dtype=np.float64)
    for values in values_by_species:
        samples += values[rng.integers(0, len(values), size=(int(repeats), len(values)))].mean(axis=1)
    samples /= len(values_by_species)
    corrections = sum(first[key]["prediction"] != first[key]["label"] and second[key]["prediction"] == second[key]["label"] for key in ids)
    regressions = sum(first[key]["prediction"] == first[key]["label"] and second[key]["prediction"] != second[key]["label"] for key in ids)
    return {
        "groups": len(ids),
        "species": len(labels),
        "budget1_macro_top1": float(np.mean([np.mean([first[key]["prediction"] == label for key in ids if first[key]["label"] == label]) for label in labels])),
        "budget2_macro_top1": float(np.mean([np.mean([second[key]["prediction"] == label for key in ids if first[key]["label"] == label]) for label in labels])),
        "delta": point,
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "corrections": int(corrections),
        "regressions": int(regressions),
        "net_corrections": int(corrections - regressions),
    }


def _metric_rows(method: str, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for budget, values in sorted(metrics["by_budget"].items(), key=lambda item: int(item[0])):
        rows.append({"method": method, "scope": "all_eligible", "budget": int(budget), **{key: values.get(key) for key in ("groups", "macro_top1", "micro_top1", "nll", "brier", "ece", "top1", "top5")}})
    rows.append({"method": method, "scope": "all_eligible", "budget": "aubc", "normalized_macro_aubc": metrics.get("normalized_macro_aubc")})
    return rows


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    run_manifest_path = resolve_config_path(config, paths["run_manifest_json"])
    summary_json = resolve_config_path(config, paths["summary_json"])
    summary_csv = resolve_config_path(config, paths["summary_csv"])
    if summary_json.exists() or summary_csv.exists():
        raise FileExistsError("P4 summary artifacts already exist")
    run_manifest = _load_json(run_manifest_path)
    if run_manifest.get("status") != "COMPLETE_P4_FULL_BENCHMARK":
        raise RuntimeError("P4 run manifest is not complete")
    p3_result_path = resolve_config_path(config, paths["p3_result_json"])
    p3_predictions_path = resolve_config_path(config, paths["p3_predictions_csv"])
    p3_result = _load_json(p3_result_path)
    if p3_result.get("status") != "STOP_P3_RISK_AWARE_GATE":
        raise RuntimeError("P4 requires the fixed P3 result, including its negative gate status")
    metrics_by_method: dict[str, dict[str, Any]] = {
        "rlra": p3_result["by_method"]["rlra"],
        "mean_logit": p3_result["by_method"]["mean_logit"],
    }
    predictions_by_method: dict[str, list[dict[str, Any]]] = {
        "rlra": _records(p3_predictions_path, "rlra"),
        "mean_logit": _records(p3_predictions_path, "mean_logit"),
    }
    for run in run_manifest["runs"]:
        method = str(run["model"])
        metrics_path = Path(str(run["metrics"]))
        predictions_path = Path(str(run["predictions"]))
        metrics_by_method[method] = _load_json(metrics_path)
        predictions_by_method[method] = _records(predictions_path)
    metric_rows: list[dict[str, Any]] = []
    pairwise: dict[str, Any] = {}
    for index, method in enumerate(metrics_by_method):
        metric_rows.extend(_metric_rows(method, metrics_by_method[method]))
        pairwise[method] = _pairwise(predictions_by_method[method], int(config["evaluation"]["bootstrap_repeats"]), int(config["seed"]) + index)
    summary = {
        "status": "COMPLETE_P4_FULL_BENCHMARK_SUMMARY",
        "claim_scope": "CONTINUE_DEVELOPMENT_ONLY",
        "methods": list(metrics_by_method),
        "p3_status": p3_result["status"],
        "metrics": metrics_by_method,
        "common_budget12": pairwise,
        "provenance": {
            "config_sha256": config["_config_hash"],
            "p3_result_sha256": sha256_file(p3_result_path),
            "p3_predictions_sha256": sha256_file(p3_predictions_path),
            "run_manifest_sha256": sha256_file(run_manifest_path),
        },
    }
    atomic_write_csv(summary_csv, pd.DataFrame.from_records(metric_rows), refuse_if_exists=True)
    summary["provenance"]["summary_csv_sha256"] = sha256_file(summary_csv)
    atomic_write_json(summary_json, summary, refuse_if_exists=True)
    print(json.dumps({"summary_json": str(summary_json), "summary_csv": str(summary_csv), "methods": list(metrics_by_method)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
