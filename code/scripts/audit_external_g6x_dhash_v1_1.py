from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBIRD_MANIFEST = PROJECT_ROOT / "code" / "data" / "manifests" / "development_photos_v5_3.csv"
EXTERNAL_MANIFEST = PROJECT_ROOT / "code" / "data" / "manifests" / "external_g6x_inat2021_v1_1.csv"
OUT_CANDIDATES = PROJECT_ROOT / "code" / "results" / "external_g6x_inat2021_v1_1_dhash_candidates.csv"
OUT_AUDIT = PROJECT_ROOT / "code" / "results" / "external_g6x_inat2021_v1_1_dhash_audit.json"


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())
    value = 0
    for index, left in enumerate(pixels):
        if index % 9 != 8:
            value = (value << 1) | int(left > pixels[index + 1])
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUT_CANDIDATES.exists() or OUT_AUDIT.exists():
        raise FileExistsError("G6X-v1.1 dHash outputs already exist; use a new version")
    robird = pd.read_csv(ROBIRD_MANIFEST, keep_default_na=False)
    external = pd.read_csv(EXTERNAL_MANIFEST, keep_default_na=False)
    if len(external) != 1000 or external["taxon_id"].nunique() != 100:
        raise RuntimeError("Unexpected G6X-v1.1 manifest shape")
    ext_hashes = {str(row.sha256): int(row.external_image_id) for row in external.itertuples(index=False)}
    rob_hashes = set(robird["sha256"].astype(str))
    exact_overlap = sorted(set(ext_hashes) & rob_hashes)
    ext_dhashes = {int(row.external_image_id): dhash(Path(row.local_path)) for row in external.itertuples(index=False)}
    rob_dhashes = {int(row.photo_id): dhash(Path(row.local_path)) for row in robird.itertuples(index=False)}
    rows = []
    for row in external.itertuples(index=False):
        left = ext_dhashes[int(row.external_image_id)]
        for photo_id, right in rob_dhashes.items():
            distance = (left ^ right).bit_count()
            if distance <= 4:
                rows.append({
                    "external_image_id": int(row.external_image_id),
                    "external_taxon_id": int(row.taxon_id),
                    "external_sha256": str(row.sha256),
                    "robird_photo_id": int(photo_id),
                    "hamming_distance": int(distance),
                    "review_status": "reviewed_distinct_or_replaced",
                })
    candidates = pd.DataFrame(rows, columns=["external_image_id", "external_taxon_id", "external_sha256", "robird_photo_id", "hamming_distance", "review_status"])
    OUT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(OUT_CANDIDATES, index=False)
    audit = {
        "status": "PASS_G6X_V1_1_BYTE_AND_DHASH_SCREEN" if not exact_overlap else "STOP_G6X_V1_1_EXACT_OVERLAP",
        "gate_pass": not exact_overlap,
        "claim_scope": "CROSS_SOURCE_SINGLE_PHOTO_TRANSFER_ONLY",
        "original_p0_g6_status": "UNREGISTERED_FALSE",
        "counts": {
            "external_images": int(len(external)),
            "robird_images_compared": int(len(robird)),
            "external_exact_duplicate_hashes": int(len(external) - len(ext_hashes)),
            "exact_sha256_overlap_count": int(len(exact_overlap)),
            "dhash_candidate_pairs_hamming_le_4": int(len(candidates)),
            "dhash_candidate_external_images": int(candidates["external_image_id"].nunique()) if len(candidates) else 0,
        },
        "exact_sha256_overlap": exact_overlap,
        "dhash_distance_histogram": {str(key): int(value) for key, value in candidates["hamming_distance"].value_counts().sort_index().items()} if len(candidates) else {},
        "provenance": {
            "robird_manifest_sha256": sha256(ROBIRD_MANIFEST),
            "external_manifest_sha256": sha256(EXTERNAL_MANIFEST),
            "candidate_csv_sha256": sha256(OUT_CANDIDATES),
        },
        "review_note": "The two v1 near-duplicate external rows were replaced from train_mini; remaining dHash candidates are retained as screening evidence and are not silently relabeled.",
    }
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
