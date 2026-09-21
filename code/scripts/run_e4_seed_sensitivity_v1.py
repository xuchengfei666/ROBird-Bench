from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from robird.budgets import stable_key
from robird.data import ObservationFeatureDataset, collate_observations
from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file
from robird.models import build_model
from robird.supplements import evaluate_manifest, load_context
from robird.training import fit, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E4 three-seed sensitivity experiments.")
    parser.add_argument("--config", type=Path, default=Path("configs/e4_seed_sensitivity_v1.yaml"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def class_map(manifest: pd.DataFrame) -> dict[int, int]:
    pairs = manifest[["taxon_id", "class_index"]].drop_duplicates()
    if pairs["taxon_id"].duplicated().any() or pairs["class_index"].duplicated().any():
        raise ValueError("Taxon/class mapping is not one-to-one")
    return {int(row.taxon_id): int(row.class_index) for row in pairs.itertuples(index=False)}


def loader(dataset: ObservationFeatureDataset, config: dict[str, Any], device: torch.device, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_observations,
    )


def records_frame(records: list[dict[str, Any]], seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    counters: dict[tuple[int, int], int] = {}
    for record in records:
        key = (int(record["observation_id"]), int(record["budget"]))
        replicate = counters.get(key, 0)
        counters[key] = replicate + 1
        photo_ids = [int(value) for value in record["photo_ids"]]
        rows.append(
            {
                "observation_id": key[0],
                "budget": key[1],
                "replicate": replicate,
                "subset_key": stable_key(seed, key[0], key[1], *photo_ids),
                "photo_ids": ";".join(str(value) for value in photo_ids),
                "label": int(record["label"]),
                "prediction": int(record["prediction"]),
                "probabilities_json": json.dumps(record["probabilities"].tolist(), separators=(",", ":")),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> int:
    args = parse_args()
    base_config = load_yaml(args.config)
    require_frozen_protocol(base_config)
    manifest, features, feature_index = load_context(base_config)
    class_indices = class_map(manifest)
    device = device_from_name(args.device)
    result_root = resolve_config_path(base_config, base_config["paths"]["result_root"])
    if result_root.exists() and any(result_root.iterdir()):
        raise FileExistsError(f"E4 result directory is non-empty: {result_root}")
    result_root.mkdir(parents=True, exist_ok=True)

    run_records: list[dict[str, Any]] = []
    models = [str(value) for value in base_config["experiment"]["models"]]
    seeds = [int(value) for value in base_config["experiment"]["seeds"]]

    # Train all six checkpoints first. Development-test is intentionally not read here.
    for model_name in models:
        for seed in seeds:
            run_name = f"{model_name.replace('_', '-')}-seed{seed}"
            run_dir = result_root / run_name
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(f"E4 output already exists: {run_dir}")
            run_dir.mkdir(parents=True, exist_ok=True)
            config = deepcopy(base_config)
            config["seed"] = seed
            config["model"] = {
                "name": model_name,
                "feature_dim": int(base_config["model"]["feature_dim"]),
                "hidden_dim": int(base_config["model"]["hidden_dim"]),
                "num_heads": int(base_config["model"]["num_heads"]),
                "num_layers": int(base_config["model"]["num_layers"]),
                "dropout": float(base_config["model"]["dropout"]),
            }
            seed_everything(seed)
            train_data = ObservationFeatureDataset(
                manifest,
                features,
                feature_index,
                split="train",
                budget=None,
                seed=seed,
                subset_mode=str(config["data"]["train_subset_mode"]),
                max_subsets=int(config["data"]["max_subsets_per_budget"]),
            )
            validation_data = ObservationFeatureDataset(
                manifest,
                features,
                feature_index,
                split="validation",
                budget=None,
                seed=seed,
                subset_mode=str(config["data"]["eval_subset_mode"]),
                max_subsets=int(config["data"]["max_subsets_per_budget"]),
            )
            model = build_model(config["model"], num_classes=len(class_indices))
            provenance = {
                "protocol": dict(base_config["protocol"]),
                "config_sha256": base_config["_config_hash"],
                "dataset_audit_sha256": sha256_file(resolve_config_path(base_config, base_config["paths"]["dataset_audit_json"])),
                "manifest_sha256": sha256_file(resolve_config_path(base_config, base_config["paths"]["manifest_csv"])),
                "splits_sha256": sha256_file(resolve_config_path(base_config, base_config["paths"]["splits_csv"])),
                "feature_index_sha256": sha256_file(resolve_config_path(base_config, base_config["paths"]["feature_index_csv"])),
                "class_map": class_indices,
                "model_config": config["model"],
                "training_budget": None,
                "run_name": run_name,
                "seed": seed,
            }
            history = fit(
                model,
                loader(train_data, config, device, shuffle=True),
                loader(validation_data, config, device, shuffle=False),
                config,
                device,
                run_dir / "best.pth",
                provenance,
            )
            atomic_write_json(run_dir / "history.json", history, refuse_if_exists=True)
            run_records.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "run_name": run_name,
                    "checkpoint": str(run_dir / "best.pth"),
                    "checkpoint_sha256": sha256_file(run_dir / "best.pth"),
                    "history": str(run_dir / "history.json"),
                    "history_sha256": sha256_file(run_dir / "history.json"),
                }
            )
            print(json.dumps({"model": model_name, "seed": seed, "stage": "TRAIN_COMPLETE"}), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Evaluate only after every checkpoint is fixed.
    metric_rows: list[dict[str, Any]] = []
    for record in run_records:
        checkpoint_path = Path(record["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint["provenance"]["model_config"]
        model = build_model(model_config, num_classes=len(class_indices))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        eval_config = deepcopy(base_config)
        eval_config["seed"] = int(record["seed"])
        summary, predictions = evaluate_manifest(model, manifest, features, feature_index, eval_config, "development_test", device)
        atomic_write_json(checkpoint_path.parent / "metrics.json", summary, refuse_if_exists=True)
        atomic_write_csv(checkpoint_path.parent / "predictions.csv", records_frame(predictions, int(record["seed"])), refuse_if_exists=True)
        record["metrics"] = str(checkpoint_path.parent / "metrics.json")
        record["metrics_sha256"] = sha256_file(checkpoint_path.parent / "metrics.json")
        record["predictions"] = str(checkpoint_path.parent / "predictions.csv")
        record["predictions_sha256"] = sha256_file(checkpoint_path.parent / "predictions.csv")
        for budget, values in summary["by_budget"].items():
            metric_rows.append(
                {
                    "model": record["model"],
                    "seed": int(record["seed"]),
                    "budget": int(budget),
                    **{key: float(value) for key, value in values.items()},
                }
            )
        print(json.dumps({"model": record["model"], "seed": record["seed"], "stage": "EVAL_COMPLETE"}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics_frame = pd.DataFrame.from_records(metric_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for (model_name, budget), group in metrics_frame.groupby(["model", "budget"], sort=True):
        for metric in ("macro_top1", "micro_top1", "top5", "nll", "brier", "ece"):
            values = group[metric].astype(float).to_numpy()
            aggregate_rows.append(
                {
                    "model": model_name,
                    "budget": int(budget),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std_sample": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    atomic_write_csv(result_root / "per_seed_metrics.csv", metrics_frame, refuse_if_exists=True)
    atomic_write_csv(result_root / "aggregate_metrics.csv", pd.DataFrame.from_records(aggregate_rows), refuse_if_exists=True)
    summary = {
        "status": "COMPLETE_E4_SEED_SENSITIVITY_V1",
        "config_sha256": base_config["_config_hash"],
        "manifest_sha256": sha256_file(resolve_config_path(base_config, base_config["paths"]["manifest_csv"])),
        "models": models,
        "seeds": seeds,
        "runs": run_records,
        "per_seed_metrics_sha256": sha256_file(result_root / "per_seed_metrics.csv"),
        "aggregate_metrics_sha256": sha256_file(result_root / "aggregate_metrics.csv"),
    }
    atomic_write_json(result_root / "summary.json", summary, refuse_if_exists=True)
    print(json.dumps({"status": summary["status"], "runs": len(run_records), "result_root": str(result_root)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
