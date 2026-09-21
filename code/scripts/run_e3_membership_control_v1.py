from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file
from robird.metrics import aggregate_subset_records
from robird.supplements import (
    build_circular_shuffle_manifest,
    correctness_by_observation,
    evaluate_manifest,
    load_checkpoint_model,
    load_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E3 same-observation membership negative control.")
    parser.add_argument("--config", type=Path, default=Path("configs/e3_membership_control_v1.yaml"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    manifest, features, feature_index = load_context(config)
    split = str(config["experiment"]["split"])
    shuffled_manifest, shuffle_meta = build_circular_shuffle_manifest(manifest, split)
    eligible_ids = set(int(value) for value in shuffled_manifest["observation_id"].unique())
    real_manifest = manifest.loc[manifest["observation_id"].isin(eligible_ids)].copy()
    device = device_from_name(args.device)
    result_root = resolve_config_path(config, config["paths"]["result_root"])
    if result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(f"E3 result directory is non-empty: {result_root}")
    result_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_root / "shuffle_assignment.json", shuffle_meta, refuse_if_exists=True)
    atomic_write_csv(result_root / "shuffled_manifest.csv", shuffled_manifest, refuse_if_exists=True)

    model_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    checkpoints = config["paths"]["checkpoints"]
    for model_name, checkpoint_value in checkpoints.items():
        checkpoint_path = resolve_config_path(config, checkpoint_value)
        model, provenance = load_checkpoint_model(checkpoint_path)
        real_summary, real_records = evaluate_manifest(
            model, real_manifest, features, feature_index, config, split, device
        )
        shuffled_summary, shuffled_records = evaluate_manifest(
            model, shuffled_manifest, features, feature_index, config, split, device
        )
        real_by_budget = {int(row["budget"]): row for row in aggregate_subset_records(real_records)}
        shuffled_by_budget = {int(row["budget"]): row for row in aggregate_subset_records(shuffled_records)}
        real_correct = correctness_by_observation(real_records)
        shuffled_correct = correctness_by_observation(shuffled_records)
        budget_comparison: dict[str, Any] = {}
        for budget in sorted(set(real_by_budget) & set(shuffled_by_budget)):
            real_rows = [row for row in aggregate_subset_records(real_records) if int(row["budget"]) == budget]
            shuffled_rows = [row for row in aggregate_subset_records(shuffled_records) if int(row["budget"]) == budget]
            real_acc = float(np.mean([int(row["prediction"]) == int(row["label"]) for row in real_rows]))
            shuffled_acc = float(np.mean([int(row["prediction"]) == int(row["label"]) for row in shuffled_rows]))
            shared = sorted(set(real_correct) & set(shuffled_correct))
            changed = sum(real_correct[key] != shuffled_correct[key] for key in shared if key[1] == budget)
            budget_comparison[str(budget)] = {
                "real_micro_top1": real_acc,
                "shuffled_micro_top1": shuffled_acc,
                "shuffled_minus_real_pp": 100.0 * (shuffled_acc - real_acc),
                "shared_observations": len([key for key in shared if key[1] == budget]),
                "correctness_changed": int(changed),
            }
            for row in real_rows:
                observation_id = int(row["observation_id"])
                shuffled_row = next(item for item in shuffled_rows if int(item["observation_id"]) == observation_id)
                comparison_rows.append(
                    {
                        "model": model_name,
                        "budget": budget,
                        "observation_id": observation_id,
                        "label": int(row["label"]),
                        "real_prediction": int(row["prediction"]),
                        "shuffled_prediction": int(shuffled_row["prediction"]),
                        "real_correct": int(int(row["prediction"]) == int(row["label"])),
                        "shuffled_correct": int(int(shuffled_row["prediction"]) == int(row["label"])),
                    }
                )
        model_results[model_name] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_provenance_config_sha256": provenance.get("config_sha256"),
            "real": real_summary,
            "shuffled": shuffled_summary,
            "comparison_by_budget": budget_comparison,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    atomic_write_csv(result_root / "paired_comparisons.csv", pd.DataFrame.from_records(comparison_rows), refuse_if_exists=True)
    result = {
        "status": "COMPLETE_E3_MEMBERSHIP_CONTROL_V1",
        "claim_scope": "SAME_TAXON_CARDINALITY_SHUFFLED_MEMBERSHIP_NEGATIVE_CONTROL",
        "config_sha256": config["_config_hash"],
        "manifest_sha256": sha256_file(resolve_config_path(config, config["paths"]["manifest_csv"])),
        "split": split,
        "eligible_observations": len(eligible_ids),
        "shuffle_assignment_sha256": sha256_file(result_root / "shuffle_assignment.json"),
        "shuffled_manifest_sha256": sha256_file(result_root / "shuffled_manifest.csv"),
        "models": model_results,
    }
    atomic_write_json(result_root / "summary.json", result, refuse_if_exists=True)
    print(json.dumps({"status": result["status"], "result_root": str(result_root), "models": len(model_results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
