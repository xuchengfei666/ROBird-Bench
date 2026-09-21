from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import timm
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from robird.io import atomic_write_csv, atomic_write_json, load_yaml, require_frozen_protocol, resolve_config_path, sha256_file


class PhotoDataset(Dataset[tuple[int, torch.Tensor]]):
    def __init__(self, manifest: pd.DataFrame, transform: Any) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        path = Path(str(self.manifest.at[index, "local_path"]))
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return index, tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen ROBird photo features.")
    parser.add_argument("--config", type=Path, default=Path("configs/features_dinov2_v5_3.yaml"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    require_frozen_protocol(config)
    paths = config["paths"]
    expected = config["expected"]
    manifest_path = resolve_config_path(config, paths["manifest_csv"])
    splits_path = resolve_config_path(config, paths["splits_csv"])
    dataset_audit_path = resolve_config_path(config, paths["dataset_audit_json"])
    split_audit_path = resolve_config_path(config, paths["split_audit_json"])
    matrix_path = resolve_config_path(config, paths["feature_matrix"])
    index_path = resolve_config_path(config, paths["feature_index_csv"])
    audit_path = resolve_config_path(config, paths["extraction_audit_json"])
    if matrix_path.exists() or index_path.exists() or audit_path.exists():
        raise FileExistsError("Frozen feature outputs already exist")
    for path, value, label in (
        (manifest_path, expected["manifest_sha256"], "manifest"),
        (splits_path, expected["splits_sha256"], "splits"),
        (dataset_audit_path, expected["dataset_audit_sha256"], "dataset_audit"),
        (split_audit_path, expected["split_audit_sha256"], "split_audit"),
    ):
        actual = sha256_file(path)
        if actual != str(value):
            raise RuntimeError(f"Frozen {label} hash mismatch: {actual}")
    dataset_audit = load_json(dataset_audit_path)
    if dataset_audit.get("decision") not in {"CONTINUE_DEVELOPMENT_ONLY", "CONTINUE_TO_COMMON_BENCHMARK"}:
        raise RuntimeError("Dataset P0 does not authorize feature extraction")
    split_audit = load_json(split_audit_path)
    if split_audit.get("validation", {}).get("passed") is not True:
        raise RuntimeError("Observer-disjoint split audit has not passed")

    manifest = pd.read_csv(manifest_path, keep_default_na=False).sort_values("row_id").reset_index(drop=True)
    if len(manifest) != int(expected["photos"]) or manifest["photo_id"].nunique() != len(manifest):
        raise RuntimeError("Unexpected manifest photo count or duplicate photo IDs")
    if manifest["observation_id"].nunique() != int(expected["groups"]) or manifest["taxon_id"].nunique() != int(expected["species"]):
        raise RuntimeError("Unexpected manifest group/species count")

    model_config = config["model"]
    model = timm.create_model(
        str(model_config["name"]), pretrained=True, num_classes=0, img_size=int(model_config["image_size"])
    )
    if int(getattr(model, "num_features")) != int(model_config["feature_dim"]):
        raise RuntimeError("Loaded encoder feature dimension differs from the frozen contract")
    state_sha256 = model_state_sha256(model)
    runtime = config["runtime"]
    if str(runtime["device"]) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Frozen feature protocol requires CUDA")
    device = torch.device(str(runtime["device"]))
    model.eval().to(device)
    interpolation = {"bicubic": InterpolationMode.BICUBIC}[str(model_config["interpolation"])]
    transform = v2.Compose(
        [
            v2.Resize(int(model_config["resize_shorter"]), interpolation=interpolation, antialias=True),
            v2.CenterCrop(int(model_config["image_size"])),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(model_config["mean"]), std=list(model_config["std"])),
        ]
    )
    loader = DataLoader(
        PhotoDataset(manifest, transform),
        batch_size=int(runtime["batch_size"]),
        shuffle=False,
        num_workers=int(runtime["num_workers"]),
        pin_memory=True,
        persistent_workers=int(runtime["num_workers"]) > 0,
    )
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = matrix_path.with_suffix(matrix_path.suffix + ".partial")
    matrix = np.lib.format.open_memmap(
        partial_path, mode="w+", dtype=np.float32, shape=(len(manifest), int(model_config["feature_dim"]))
    )
    rows_written = 0
    with torch.inference_mode():
        for batch_index, (indices, images) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(runtime["autocast_fp16"])):
                features = model(images)
            features = features.float()
            norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
            if not torch.isfinite(features).all() or (norms <= 0).any():
                raise RuntimeError(f"Invalid feature values at batch {batch_index}")
            if bool(model_config["l2_normalize"]):
                features = features / norms
            values = features.cpu().numpy().astype(np.float32, copy=False)
            positions = indices.numpy().astype(np.int64)
            matrix[positions] = values
            rows_written += len(positions)
            if batch_index % int(runtime["checkpoint_every_batches"]) == 0:
                matrix.flush()
                print(f"features={rows_written}/{len(manifest)}", flush=True)
    matrix.flush()
    del matrix
    os.replace(partial_path, matrix_path)
    index = manifest[["photo_id", "observation_id", "observer_id", "taxon_id", "class_index", "cohort"]].copy()
    index.insert(0, "feature_row", np.arange(len(index), dtype=np.int64))
    atomic_write_csv(index_path, index, refuse_if_exists=True)
    loaded = np.load(matrix_path, mmap_mode="r")
    finite = bool(np.isfinite(loaded).all())
    norms = np.linalg.norm(loaded, axis=1)
    gates = {
        "shape": tuple(loaded.shape) == (len(manifest), int(model_config["feature_dim"])),
        "finite": finite,
        "normalized": bool(np.allclose(norms, 1.0, atol=2e-4)) if bool(model_config["l2_normalize"]) else bool((norms > 0).all()),
        "index_rows": len(index) == len(manifest),
        "photo_ids_match": set(index["photo_id"].astype(int)) == set(manifest["photo_id"].astype(int)),
        "feature_rows_contiguous": index["feature_row"].astype(int).tolist() == list(range(len(index))),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Feature extraction gates failed: {gates}")
    audit = {
        "status": "PASS_FEATURE_P0_V5_3",
        "gate_pass": True,
        "development_only": dataset_audit.get("decision") == "CONTINUE_DEVELOPMENT_ONLY",
        "model": {
            "name": str(model_config["name"]),
            "state_sha256": state_sha256,
            "feature_dim": int(model_config["feature_dim"]),
            "l2_normalize": bool(model_config["l2_normalize"]),
        },
        "software": {"torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__},
        "counts": {"photos": len(manifest), "groups": int(manifest["observation_id"].nunique()), "species": int(manifest["taxon_id"].nunique())},
        "provenance": {
            "config_sha256": config["_config_hash"],
            "manifest_sha256": sha256_file(manifest_path),
            "splits_sha256": sha256_file(splits_path),
            "dataset_audit_sha256": sha256_file(dataset_audit_path),
            "split_audit_sha256": sha256_file(split_audit_path),
            "feature_matrix_sha256": sha256_file(matrix_path),
            "feature_index_sha256": sha256_file(index_path),
        },
        "gates": gates,
    }
    atomic_write_json(audit_path, audit, refuse_if_exists=True)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
