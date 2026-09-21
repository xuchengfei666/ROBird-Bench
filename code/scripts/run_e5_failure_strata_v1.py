from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file
from robird.metrics import aggregate_subset_records, brier_score, expected_calibration_error, negative_log_likelihood
from robird.supplements import merge_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E5 ROBird failure-stratum and G6X taxon analysis.")
    parser.add_argument("--config", type=Path, default=Path("configs/e5_failure_strata_v1.yaml"))
    return parser.parse_args()


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"observation_id", "budget", "label", "prediction", "probabilities_json"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction file missing columns {missing}: {path}")
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        probabilities = np.asarray(json.loads(str(row.probabilities_json)), dtype=np.float64)
        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError(f"Invalid probability vector in {path}")
        records.append(
            {
                "observation_id": int(row.observation_id),
                "budget": int(row.budget),
                "label": int(row.label),
                "prediction": int(row.prediction),
                "probabilities": probabilities,
            }
        )
    return records


def observation_metadata(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = manifest.loc[manifest["split"] == split].copy()
    observer_counts = selected[["observation_id", "observer_id"]].drop_duplicates().groupby("observer_id").size()
    grouped = selected.groupby("observation_id", sort=True)
    rows: list[dict[str, Any]] = []
    for observation_id, group in grouped:
        taxon_ids = group["taxon_id"].unique()
        labels = group["class_index"].unique()
        if len(taxon_ids) != 1 or len(labels) != 1:
            raise ValueError(f"Observation metadata is inconsistent: {observation_id}")
        observer_id = int(group["observer_id"].iloc[0])
        area = float((group["width"].astype(float) * group["height"].astype(float)).mean())
        rows.append(
            {
                "observation_id": int(observation_id),
                "taxon_id": int(taxon_ids[0]),
                "class_index": int(labels[0]),
                "scientific_name": str(group["scientific_name"].iloc[0]),
                "observer_id": observer_id,
                "photo_count": int(len(group)),
                "observer_frequency": int(observer_counts.loc[observer_id]),
                "mean_area": area,
                "area_stratum": "<0.5M" if area < 500000 else ("0.5-1.5M" if area < 1500000 else ">=1.5M"),
                "cardinality_stratum": str(int(len(group))),
                "observer_stratum": "1" if int(observer_counts.loc[observer_id]) == 1 else ("2-3" if int(observer_counts.loc[observer_id]) <= 3 else "4+"),
            }
        )
    return pd.DataFrame.from_records(rows)


def add_prediction_fields(aggregated: list[dict[str, Any]], metadata: pd.DataFrame) -> list[dict[str, Any]]:
    lookup = metadata.set_index("observation_id").to_dict(orient="index")
    output: list[dict[str, Any]] = []
    for row in aggregated:
        observation_id = int(row["observation_id"])
        if observation_id not in lookup:
            continue
        meta = lookup[observation_id]
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        output.append(
            {
                **{key: value for key, value in meta.items()},
                "observation_id": observation_id,
                "budget": int(row["budget"]),
                "label": int(row["label"]),
                "prediction": int(row["prediction"]),
                "correct": int(int(row["prediction"]) == int(row["label"])),
                "confidence": float(probabilities.max()),
                "true_probability": float(probabilities[int(row["label"])]),
                "nll": float(-np.log(np.clip(probabilities[int(row["label"])], 1e-12, 1.0))),
            }
        )
    return output


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"groups": 0, "accuracy": None, "mean_confidence": None, "mean_true_probability": None, "nll": None}
    return {
        "groups": len(rows),
        "accuracy": float(np.mean([row["correct"] for row in rows])),
        "mean_confidence": float(np.mean([row["confidence"] for row in rows])),
        "mean_true_probability": float(np.mean([row["true_probability"] for row in rows])),
        "nll": float(np.mean([row["nll"] for row in rows])),
    }


def build_strata(rows: list[dict[str, Any]], model: str, budgets: list[int]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    definitions = {
        "cardinality": "cardinality_stratum",
        "observer_frequency": "observer_stratum",
        "mean_area": "area_stratum",
        "taxon": "scientific_name",
    }
    by_key = {(int(row["observation_id"]), int(row["budget"])): row for row in rows}
    for stratum_type, field in definitions.items():
        for stratum_value in sorted({str(row[field]) for row in rows}):
            for budget in budgets:
                selected = [row for row in rows if int(row["budget"]) == budget and str(row[field]) == stratum_value]
                summary = summarize_rows(selected)
                pairs = [
                    (by_key[(int(row["observation_id"]), 1)], by_key[(int(row["observation_id"]), 2)])
                    for row in selected
                    if budget == 2 and (int(row["observation_id"]), 1) in by_key and (int(row["observation_id"]), 2) in by_key
                ]
                corrections = sum(int(a["correct"]) == 0 and int(b["correct"]) == 1 for a, b in pairs)
                regressions = sum(int(a["correct"]) == 1 and int(b["correct"]) == 0 for a, b in pairs)
                output.append(
                    {
                        "model": model,
                        "stratum_type": stratum_type,
                        "stratum": stratum_value,
                        "budget": budget,
                        **summary,
                        "corrections_1_to_2": int(corrections) if budget == 2 else None,
                        "regressions_1_to_2": int(regressions) if budget == 2 else None,
                        "net_corrections_1_to_2": int(corrections - regressions) if budget == 2 else None,
                    }
                )
    return output


def load_external_taxon_metrics(manifest_path: Path, predictions_path: Path) -> list[dict[str, Any]]:
    manifest = pd.read_csv(manifest_path)
    predictions = pd.read_csv(predictions_path)
    required = {"taxon_id", "class_index", "method", "correct", "confidence", "true_probability"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"External predictions missing columns {missing}")
    joined = predictions.merge(
        manifest[["external_image_id", "taxon_id", "scientific_name", "width", "height"]],
        left_on=["taxon_id", "external_row_id"],
        right_on=["taxon_id", "external_image_id"],
        how="left",
    )
    if joined["scientific_name"].isna().any():
        joined = predictions.merge(manifest[["external_row_id", "taxon_id", "scientific_name", "width", "height"]], on=["external_row_id", "taxon_id"], how="left")
    output: list[dict[str, Any]] = []
    for (method, taxon_id, scientific_name), group in joined.groupby(["method", "taxon_id", "scientific_name"], dropna=False, sort=True):
        output.append(
            {
                "method": str(method),
                "taxon_id": int(taxon_id),
                "scientific_name": str(scientific_name),
                "images": int(len(group)),
                "accuracy": float(group["correct"].astype(bool).mean()),
                "mean_confidence": float(group["confidence"].mean()),
                "mean_true_probability": float(group["true_probability"].mean()),
                "mean_area": float((group["width"].astype(float) * group["height"].astype(float)).mean()),
            }
        )
    return output


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    manifest_path = resolve_config_path(config, paths["manifest_csv"])
    split_path = resolve_config_path(config, paths["splits_csv"])
    audit_path = resolve_config_path(config, paths["dataset_audit_json"])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("decision") not in {"CONTINUE_DEVELOPMENT_ONLY", "CONTINUE_TO_COMMON_BENCHMARK"}:
        raise RuntimeError(f"Unexpected dataset decision: {audit.get('decision')}")
    manifest = merge_splits(pd.read_csv(manifest_path), pd.read_csv(split_path))
    split = str(config["experiment"]["split"])
    metadata = observation_metadata(manifest, split)
    result_root = resolve_config_path(config, paths["result_root"])
    if result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(f"E5 result directory is non-empty: {result_root}")
    result_root.mkdir(parents=True, exist_ok=True)

    strata_rows: list[dict[str, Any]] = []
    robird_summary: dict[str, Any] = {}
    for model_name, prediction_value in paths["robird_predictions"].items():
        prediction_path = resolve_config_path(config, prediction_value)
        records = load_prediction_records(prediction_path)
        rows = add_prediction_fields(aggregate_subset_records(records), metadata)
        strata_rows.extend(build_strata(rows, model_name, [int(value) for value in config["experiment"]["budgets"]]))
        robird_summary[model_name] = {
            "prediction_sha256": sha256_file(prediction_path),
            "rows": len(rows),
            "observations": len({int(row["observation_id"]) for row in rows}),
        }
    external_manifest = resolve_config_path(config, paths["external_manifest"])
    external_predictions = resolve_config_path(config, paths["external_predictions"])
    external_rows = load_external_taxon_metrics(external_manifest, external_predictions)

    atomic_write_csv(result_root / "robird_strata.csv", pd.DataFrame.from_records(strata_rows), refuse_if_exists=True)
    atomic_write_csv(result_root / "external_taxon_metrics.csv", pd.DataFrame.from_records(external_rows), refuse_if_exists=True)
    result = {
        "status": "COMPLETE_E5_FAILURE_STRATA_V1",
        "claim_scope": "ROBIRD_FAILURE_STRATA_AND_G6X_SINGLE_PHOTO_TAXON_ANALYSIS",
        "config_sha256": config["_config_hash"],
        "manifest_sha256": sha256_file(manifest_path),
        "split_sha256": sha256_file(split_path),
        "robird": robird_summary,
        "external_manifest_sha256": sha256_file(external_manifest),
        "external_predictions_sha256": sha256_file(external_predictions),
        "robird_strata_sha256": sha256_file(result_root / "robird_strata.csv"),
        "external_taxon_metrics_sha256": sha256_file(result_root / "external_taxon_metrics.csv"),
        "robird_groups_in_split": int(len(metadata)),
        "external_taxa": int(len(set(row["taxon_id"] for row in external_rows))),
    }
    atomic_write_json(result_root / "summary.json", result, refuse_if_exists=True)
    print(json.dumps({"status": result["status"], "robird_strata_rows": len(strata_rows), "external_rows": len(external_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
