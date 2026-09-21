from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from robird.metrics import (
    brier_score,
    expected_calibration_error,
    macro_top1,
    negative_log_likelihood,
    softmax_numpy,
    topk_accuracy,
)
from robird.models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "code" / "data" / "manifests" / "external_g6x_inat2021_v1_1.csv"
FEATURE_MATRIX = Path("E:/Datasets/ROBird-Bench/external_g6x_v1_1/features/dinov2_vits14_224.npy")
FEATURE_INDEX = Path("E:/Datasets/ROBird-Bench/external_g6x_v1_1/features/index.csv")
OUT_ROOT = PROJECT_ROOT / "code" / "results" / "external_g6x_v1_1"
OUT_SUMMARY = OUT_ROOT / "summary.json"
OUT_CSV = OUT_ROOT / "predictions.csv"

CHECKPOINTS = {
    "mean_feature_k1": PROJECT_ROOT / "code" / "results" / "benchmark_v5_3_p1" / "mean-feature-k1" / "best.pth",
    "mean_feature_all": PROJECT_ROOT / "code" / "results" / "benchmark_v5_3_p1" / "mean-feature-all" / "best.pth",
    "deepsets_all": PROJECT_ROOT / "code" / "results" / "benchmark_v5_3_p4_1" / "deepsets-all" / "best.pth",
    "set_transformer_all": PROJECT_ROOT / "code" / "results" / "benchmark_v5_3_p4_1" / "set-transformer-all" / "best.pth",
    "rcca_all": PROJECT_ROOT / "code" / "results" / "benchmark_v5_3_p4_1" / "rcca-all" / "best.pth",
    "max_feature_all": PROJECT_ROOT / "code" / "results" / "benchmark_v5_3_p4_1" / "max-feature-all" / "best.pth",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    by_taxon = {}
    for label in sorted(set(labels.tolist())):
        selected = labels == label
        by_taxon[str(int(label))] = {
            "n": int(selected.sum()),
            "top1": float(np.mean(predictions[selected] == label)),
        }
    return {
        "images": int(len(labels)),
        "taxa": int(len(set(labels.tolist()))),
        "macro_top1": macro_top1(probabilities, labels),
        "micro_top1": topk_accuracy(probabilities, labels, 1),
        "top5": topk_accuracy(probabilities, labels, 5),
        "nll": negative_log_likelihood(probabilities, labels),
        "brier": brier_score(probabilities, labels),
        "ece": expected_calibration_error(probabilities, labels, bins=15),
        "per_taxon": by_taxon,
    }


def main() -> int:
    if OUT_SUMMARY.exists() or OUT_CSV.exists():
        raise FileExistsError("G6X evaluation outputs already exist; use a new version")
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    matrix = np.load(FEATURE_MATRIX, mmap_mode="r")
    index = pd.read_csv(FEATURE_INDEX)
    if len(manifest) != 1000 or matrix.shape != (1000, 384) or len(index) != 1000:
        raise RuntimeError("Unexpected G6X feature/manifest shape")
    if index["feature_row"].astype(int).tolist() != list(range(1000)):
        raise RuntimeError("External feature rows are not contiguous")
    labels = manifest["class_index"].astype(int).to_numpy()
    features = np.asarray(matrix, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outputs: list[pd.DataFrame] = []
    summary: dict[str, Any] = {
        "status": "COMPLETE_G6X_EXTERNAL_TRANSFER_V1_1",
        "claim_scope": "CROSS_SOURCE_SINGLE_PHOTO_TRANSFER_ONLY",
        "original_p0_g6_status": "UNREGISTERED_FALSE",
        "counts": {"images": 1000, "taxa": 100},
        "provenance": {
            "manifest_sha256": sha256_file(MANIFEST),
            "feature_matrix_sha256": sha256_file(FEATURE_MATRIX),
            "feature_index_sha256": sha256_file(FEATURE_INDEX),
            "checkpoints": {},
        },
        "methods": {},
    }
    for method, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        provenance = checkpoint.get("provenance", {})
        model_config = dict(provenance["model_config"])
        class_map = {int(key): int(value) for key, value in provenance["class_map"].items()}
        if len(class_map) != 100:
            raise RuntimeError(f"Checkpoint {method} does not contain the ROBird 100-class map")
        model = build_model(model_config, num_classes=100).to(device).eval()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        logits_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(features), 128):
                batch = torch.from_numpy(features[start : start + 128]).to(device)
                batch = batch.unsqueeze(1)
                mask = torch.ones((len(batch), 1), dtype=torch.bool, device=device)
                logits = model(batch, mask).logits
                logits_chunks.append(logits.detach().cpu().numpy())
        logits_np = np.concatenate(logits_chunks, axis=0)
        probabilities = softmax_numpy(logits_np)
        metrics = summarize(probabilities, labels)
        summary["methods"][method] = {
            "metrics": metrics,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "model_config": model_config,
            "training_budget": provenance.get("training_budget"),
            "seed": provenance.get("seed"),
        }
        top5 = np.argsort(-probabilities, axis=1)[:, :5]
        frame = manifest[["external_row_id", "external_image_id", "taxon_id", "class_index", "common_name", "local_path"]].copy()
        frame["method"] = method
        frame["prediction"] = predictions = probabilities.argmax(axis=1)
        frame["correct"] = predictions == labels
        frame["confidence"] = probabilities.max(axis=1)
        frame["true_probability"] = probabilities[np.arange(len(labels)), labels]
        frame["top5"] = [";".join(str(int(value)) for value in row) for row in top5]
        outputs.append(frame)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.concat(outputs, ignore_index=True).to_csv(OUT_CSV, index=False)
    summary["provenance"]["predictions_sha256"] = sha256_file(OUT_CSV)
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
