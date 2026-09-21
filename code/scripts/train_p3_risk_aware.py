from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from robird.data import ObservationFeatureDataset, ObservationBatch, collate_observations, load_feature_table
from robird.io import atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file
from robird.risk_aware import RiskAwareReliabilityAggregator, load_classifier_state
from robird.splits import validate_splits
from robird.training import seed_everything, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen v5.3 P3 RLRA model.")
    parser.add_argument("--config", type=Path, default=Path("configs/p3_risk_aware_v5_3.yaml"))
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
    if pairs["taxon_id"].duplicated().any() or pairs["class_index"].duplicated().any():
        raise ValueError("Taxon/class mapping is not one-to-one")
    values = sorted(pairs["class_index"].astype(int).tolist())
    if values != list(range(len(values))):
        raise ValueError("class_index values must be contiguous from zero")
    return len(values)


def _loader(dataset: ObservationFeatureDataset, batch_size: int, device: torch.device, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_observations,
    )


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _validate_frozen_inputs(config: Mapping[str, Any]) -> dict[str, Path]:
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
    hash_pairs = {
        "manifest_csv": expected["manifest_sha256"],
        "splits_csv": expected["splits_sha256"],
        "dataset_audit_json": expected["dataset_audit_sha256"],
        "split_audit_json": expected["split_audit_sha256"],
        "feature_matrix": expected["feature_matrix_sha256"],
        "feature_index_csv": expected["feature_index_sha256"],
        "feature_audit_json": expected["feature_audit_sha256"],
        "p1_checkpoint": expected["p1_checkpoint_sha256"],
        "p1_gate_audit_json": expected["p1_gate_audit_sha256"],
        "p2_result_json": expected["p2_result_sha256"],
    }
    for key, expected_hash in hash_pairs.items():
        actual = sha256_file(resolved[key])
        if actual != str(expected_hash):
            raise RuntimeError(f"Frozen {key} hash mismatch: {actual}")
    dataset_audit = _load_json(resolved["dataset_audit_json"])
    split_audit = _load_json(resolved["split_audit_json"])
    p1_gate = _load_json(resolved["p1_gate_audit_json"])
    p2_result = _load_json(resolved["p2_result_json"])
    if dataset_audit.get("decision") != "CONTINUE_DEVELOPMENT_ONLY":
        raise RuntimeError("P3 requires the v5.3 development-only Dataset P0 decision")
    if split_audit.get("validation", {}).get("passed") is not True:
        raise RuntimeError("P3 requires a passing observer-disjoint split audit")
    if p1_gate.get("status") != "STOP_P1_TASK_SIGNAL_GATE":
        raise RuntimeError("P3 input P1 evidence is not the frozen task-signal STOP")
    if p2_result.get("status") != "STOP_P2_HARMFUL_VIEW_GATE":
        raise RuntimeError("P3 input P2 evidence is not the frozen harmful-view STOP")
    return resolved


def _step_loss(
    model: RiskAwareReliabilityAggregator,
    features: Tensor,
    mask: Tensor,
    labels: Tensor,
    criterion: nn.Module,
    risk_samples: int,
    risk_quantile: float,
    risk_weight: float,
    consistency_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    deterministic = model(features, mask, stochastic=False)
    full_losses = criterion(deterministic.logits, labels)
    dropout_losses: list[Tensor] = []
    consistency_losses: list[Tensor] = []
    reference = torch.softmax(deterministic.logits.detach(), dim=1)
    for _ in range(int(risk_samples)):
        stochastic = model(features, mask, stochastic=True)
        dropout_losses.append(criterion(stochastic.logits, labels))
        consistency_losses.append(
            F.kl_div(
                F.log_softmax(stochastic.logits, dim=1),
                reference,
                reduction="batchmean",
            )
        )
    stacked = torch.stack(dropout_losses, dim=0)
    tail_count = max(1, int(np.ceil(float(risk_quantile) * int(risk_samples))))
    cvar = stacked.topk(tail_count, dim=0).values.mean()
    consistency = torch.stack(consistency_losses).mean()
    loss = full_losses.mean() + float(risk_weight) * cvar + float(consistency_weight) * consistency
    return loss, {
        "full_ce": float(full_losses.detach().mean()),
        "cvar": float(cvar.detach()),
        "consistency": float(consistency.detach()),
        "loss": float(loss.detach()),
    }


def main() -> int:
    args = parse_args()
    config = _load_yaml_and_require(args.config)
    resolved = _validate_frozen_inputs(config)
    output_checkpoint = resolve_config_path(config, config["paths"]["checkpoint"])
    history_path = resolve_config_path(config, config["paths"]["history_json"])
    if output_checkpoint.exists() or history_path.exists():
        raise FileExistsError("P3 training artifacts already exist")
    manifest = _merge_splits(pd.read_csv(resolved["manifest_csv"]), pd.read_csv(resolved["splits_csv"]))
    features, feature_index = load_feature_table(resolved["feature_matrix"], resolved["feature_index_csv"])
    class_count = _class_count(manifest)
    checkpoint = torch.load(resolved["p1_checkpoint"], map_location="cpu")
    model_config = config["model"]
    model = RiskAwareReliabilityAggregator(
        feature_dim=int(model_config["feature_dim"]),
        num_classes=class_count,
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        keep_probability=float(model_config["keep_probability"]),
        anchor_probability=float(model_config["anchor_probability"]),
    )
    load_classifier_state(model, checkpoint)
    seed_everything(int(config["seed"]))
    device = _device(args.device)
    data_config = config["data"]
    training = config["training"]
    train_full = ObservationFeatureDataset(
        manifest, features, feature_index, "train", None, int(config["seed"]), "first", int(data_config["max_subsets_per_budget"])
    )
    validation_full = ObservationFeatureDataset(
        manifest, features, feature_index, "validation", None, int(config["seed"]), "exhaustive", int(data_config["max_subsets_per_budget"])
    )
    validation_k1 = ObservationFeatureDataset(
        manifest, features, feature_index, "validation", 1, int(config["seed"]), "exhaustive", int(data_config["max_subsets_per_budget"])
    )
    validation_k2 = ObservationFeatureDataset(
        manifest, features, feature_index, "validation", 2, int(config["seed"]), "exhaustive", int(data_config["max_subsets_per_budget"])
    )
    train_loader = _loader(train_full, int(training["batch_size"]), device, True)
    validation_loaders = {
        "full": _loader(validation_full, int(training["batch_size"]), device, False),
        "budget1": _loader(validation_k1, int(training["batch_size"]), device, False),
        "budget2": _loader(validation_k2, int(training["batch_size"]), device, False),
    }
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": float(training["learning_rate"]) * float(training["base_learning_rate_scale"])},
            {"params": [p for name, p in model.named_parameters() if not name.startswith("classifier.")]},
        ],
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training["label_smoothing"]), reduction="none")
    weights = config.get("validation_score_weights", {"full": 0.25, "budget1": 0.25, "budget2": 0.5})
    best_score = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    provenance = {
        "protocol": dict(config["protocol"]),
        "config_sha256": config["_config_hash"],
        "dataset_audit_sha256": sha256_file(resolved["dataset_audit_json"]),
        "manifest_sha256": sha256_file(resolved["manifest_csv"]),
        "splits_sha256": sha256_file(resolved["splits_csv"]),
        "feature_matrix_sha256": sha256_file(resolved["feature_matrix"]),
        "feature_index_sha256": sha256_file(resolved["feature_index_csv"]),
        "p1_checkpoint_sha256": sha256_file(resolved["p1_checkpoint"]),
        "p2_result_sha256": sha256_file(resolved["p2_result_json"]),
        "model_config": {**dict(model_config), "num_classes": class_count},
        "seed": int(config["seed"]),
        "method": "RLRA",
    }
    for epoch in range(int(training["epochs"])):
        model.train()
        totals = {"full_ce": 0.0, "cvar": 0.0, "consistency": 0.0, "loss": 0.0}
        count = 0
        for batch in train_loader:
            features_batch = batch.features.to(device, non_blocking=True)
            mask_batch = batch.mask.to(device, non_blocking=True)
            labels_batch = batch.labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, values = _step_loss(
                model, features_batch, mask_batch, labels_batch, criterion,
                int(training["risk_samples"]), float(training["risk_quantile"]),
                float(training["risk_weight"]), float(training["consistency_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key in totals:
                totals[key] += values[key] * len(labels_batch)
            count += len(labels_batch)
        validation = {name: validate(model, loader, device) for name, loader in validation_loaders.items()}
        score = sum(float(weights[name]) * float(validation[name]["loss"]) for name in ("full", "budget1", "budget2"))
        row = {
            "epoch": epoch,
            "train": {key: value / max(count, 1) for key, value in totals.items()},
            "validation": validation,
            "selection_score": score,
        }
        history.append(row)
        if score < best_score:
            best_score = score
            stale = 0
            _atomic_torch_save(
                output_checkpoint,
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "selection_score": score,
                    "provenance": provenance,
                },
            )
        else:
            stale += 1
            if stale >= int(training["patience"]):
                break
        print(json.dumps({"epoch": epoch, **row["train"], "selection_score": score}, sort_keys=True), flush=True)
    atomic_write_json(history_path, {"history": history, "best_selection_score": best_score}, refuse_if_exists=True)
    print(json.dumps({"checkpoint": str(output_checkpoint), "history": str(history_path), "epochs": len(history)}, indent=2))
    return 0


def _load_yaml_and_require(path: Path) -> dict[str, Any]:
    config = load_yaml(path)
    require_frozen_protocol(config)
    return config


if __name__ == "__main__":
    raise SystemExit(main())
