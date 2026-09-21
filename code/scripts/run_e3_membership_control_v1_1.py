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
from robird.supplements import correctness_by_observation, evaluate_manifest, load_checkpoint_model, load_context
from robird.supplements_e3_v1_1 import build_photo_level_shuffle_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected E3 photo-level membership control.")
    parser.add_argument("--config", type=Path, default=Path("configs/e3_membership_control_v1_1.yaml"))
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
    shuffled_manifest, shuffle_meta = build_photo_level_shuffle_manifest(manifest, split)
    eligible_ids = set(int(value) for value in shuffled_manifest["observation_id"].unique())
    real_manifest = manifest.loc[manifest["observation_id"].isin(eligible_ids)].copy()
    device = device_from_name(args.device)
    result_root = resolve_config_path(config, config["paths"]["result_root"])
    if result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(f"E3-v1.1 result directory is non-empty: {result_root}")
    result_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_root / "shuffle_assignment.json", shuffle_meta, refuse_if_exists=True)
    atomic_write_csv(result_root / "shuffled_manifest.csv", shuffled_manifest, refuse_if_exists=True)

    model_results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for model_name, checkpoint_value in config["paths"]["checkpoints"].items():
        checkpoint_path = resolve_config_path(config, checkpoint_value)
        model, provenance = load_checkpoint_model(checkpoint_path)
        real_summary, real_records = evaluate_manifest(model, real_manifest, features, feature_index, config, split, device)
        shuffled_summary, shuffled_records = evaluate_manifest(model, shuffled_manifest, features, feature_index, config, split, device)
        real_aggregated = aggregate_subset_records(real_records)
        shuffled_aggregated = aggregate_subset_records(shuffled_records)
        real_by_key = {(int(row["observation_id"]), int(row["budget"])): row for row in real_aggregated}
        shuffled_by_key = {(int(row["observation_id"]), int(row["budget"])): row for row in shuffled_aggregated}
        real_correct = correctness_by_observation(real_records)
        shuffled_correct = correctness_by_observation(shuffled_records)
        budget_comparison: dict[str, Any] = {}
        for budget in sorted({int(row["budget"]) for row in real_aggregated}):
            keys = sorted(key for key in real_by_key if key[1] == budget and key in shuffled_by_key)
            if not keys:
                continue
            real_acc = float(np.mean([real_correct[key] for key in keys]))
            shuffled_acc = float(np.mean([shuffled_correct[key] for key in keys]))
            changed = sum(real_correct[key] != shuffled_correct[key] for key in keys)
            budget_comparison[str(budget)] = {
                "shared_observations": len(keys),
                "real_micro_top1": real_acc,
                "shuffled_micro_top1": shuffled_acc,
                "shuffled_minus_real_pp": 100.0 * (shuffled_acc - real_acc),
                "correctness_changed": int(changed),
            }
            for key in keys:
                real_row = real_by_key[key]
                shuffled_row = shuffled_by_key[key]
                comparison_rows.append(
                    {
                        "model": model_name,
                        "budget": budget,
                        "observation_id": key[0],
                        "label": int(real_row["label"]),
                        "real_prediction": int(real_row["prediction"]),
                        "shuffled_prediction": int(shuffled_row["prediction"]),
                        "real_correct": int(real_correct[key]),
                        "shuffled_correct": int(shuffled_correct[key]),
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
        "status": "COMPLETE_E3_MEMBERSHIP_CONTROL_V1_1",
        "claim_scope": "PHOTO_LEVEL_SAME_TAXON_CARDINALITY_MEMBERSHIP_NEGATIVE_CONTROL",
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
