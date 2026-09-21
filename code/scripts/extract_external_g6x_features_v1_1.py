from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "code" / "data" / "manifests" / "external_g6x_inat2021_v1_1.csv"
FEATURE_ROOT = Path("E:/Datasets/ROBird-Bench/external_g6x_v1_1/features")
MATRIX = FEATURE_ROOT / "dinov2_vits14_224.npy"
INDEX = FEATURE_ROOT / "index.csv"
AUDIT = PROJECT_ROOT / "code" / "results" / "external_g6x_inat2021_v1_1_feature_audit.json"


class PhotoDataset(Dataset[tuple[int, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, transform: object) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        with Image.open(Path(str(self.frame.at[index, "local_path"]))) as image:
            return index, self.transform(image.convert("RGB"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if any(path.exists() for path in (MATRIX, INDEX, AUDIT)):
        raise FileExistsError("G6X feature outputs already exist; use a new version")
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    if len(manifest) != 1000 or manifest["taxon_id"].nunique() != 100:
        raise RuntimeError("Unexpected G6X-v1.1 manifest shape")
    if not all(Path(str(value)).exists() for value in manifest["local_path"]):
        raise FileNotFoundError("At least one external image path is missing")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("G6X feature protocol requires CUDA to match ROBird DINOv2 extraction")
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
    if int(model.num_features) != 384:
        raise RuntimeError("Unexpected DINOv2 feature dimension")
    state_hash = model_state_sha256(model)
    model.eval().to(device)
    transform = v2.Compose(
        [
            v2.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(PhotoDataset(manifest, transform), batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    partial = MATRIX.with_suffix(MATRIX.suffix + ".partial")
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(manifest), 384))
    written = 0
    with torch.inference_mode():
        for batch_no, (indices, images) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                features = model(images).float()
            norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
            if not torch.isfinite(features).all() or (norms <= 0).any():
                raise RuntimeError(f"Invalid features at batch {batch_no}")
            features = features / norms
            matrix[indices.numpy()] = features.cpu().numpy().astype(np.float32, copy=False)
            written += len(indices)
            if batch_no % 5 == 0:
                matrix.flush()
                print(f"features={written}/{len(manifest)}", flush=True)
    matrix.flush()
    del matrix
    partial.replace(MATRIX)
    index = manifest[["external_image_id", "group_id", "taxon_id", "class_index", "source_dataset"]].copy()
    index.insert(0, "feature_row", np.arange(len(index), dtype=np.int64))
    index.to_csv(INDEX, index=False)
    loaded = np.load(MATRIX, mmap_mode="r")
    norms = np.linalg.norm(loaded, axis=1)
    gates = {
        "shape": tuple(loaded.shape) == (1000, 384),
        "finite": bool(np.isfinite(loaded).all()),
        "normalized": bool(np.allclose(norms, 1.0, atol=2e-4)),
        "index_rows": len(index) == 1000,
        "row_contiguous": index["feature_row"].astype(int).tolist() == list(range(1000)),
        "external_ids_unique": index["external_image_id"].is_unique,
    }
    if not all(gates.values()):
        raise RuntimeError(f"G6X feature gates failed: {gates}")
    audit = {
        "status": "PASS_G6X_FEATURES_V1_1",
        "gate_pass": True,
        "claim_scope": "CROSS_SOURCE_SINGLE_PHOTO_TRANSFER_ONLY",
        "counts": {"images": 1000, "taxa": 100},
        "model": {"name": "vit_small_patch14_dinov2.lvd142m", "feature_dim": 384, "state_sha256": state_hash},
        "software": {"torch": torch.__version__, "torchvision": torchvision.__version__, "timm": timm.__version__},
        "provenance": {"manifest_sha256": sha256_file(MANIFEST), "matrix_sha256": sha256_file(MATRIX), "index_sha256": sha256_file(INDEX)},
        "paths": {"matrix": str(MATRIX), "index": str(INDEX), "manifest": str(MANIFEST)},
        "gates": gates,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
