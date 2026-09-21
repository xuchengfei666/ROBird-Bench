from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_MANIFEST = PROJECT_ROOT / "code" / "data" / "manifests" / "external_g6x_inat2021_v1.csv"
ROBIRD_MANIFEST = PROJECT_ROOT / "code" / "data" / "manifests" / "development_photos_v5_3.csv"
REVIEW = PROJECT_ROOT / "code" / "results" / "external_g6x_inat2021_v1_dhash_review.csv"
TRAIN_METADATA = Path("E:/Datasets/iNaturalist-2021/metadata/train_mini.json")
TRAIN_IMAGE_ROOT = Path("E:/Datasets/iNaturalist-2021/bird_images/train")
OUT_MANIFEST = PROJECT_ROOT / "code" / "data" / "manifests" / "external_g6x_inat2021_v1_1.csv"
OUT_AUDIT = PROJECT_ROOT / "code" / "results" / "external_g6x_inat2021_v1_1_audit.json"
OUT_MAPPING = PROJECT_ROOT / "code" / "results" / "external_g6x_inat2021_v1_1_taxonomy_mapping.csv"


ALIASES = {
    "Urile pelagicus": "Phalacrocorax pelagicus",
    "Hesperiphona vespertina": "Coccothraustes vespertinus",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8))
        pixels = list(image.getdata())
    value = 0
    for index, left in enumerate(pixels):
        if index % 9 == 8:
            continue
        value = (value << 1) | int(left > pixels[index + 1])
    return value


def main() -> int:
    if any(path.exists() for path in (OUT_MANIFEST, OUT_AUDIT, OUT_MAPPING)):
        raise FileExistsError("G6X-v1.1 outputs already exist; use a new version")
    base = pd.read_csv(BASE_MANIFEST, keep_default_na=False)
    robird = pd.read_csv(ROBIRD_MANIFEST, keep_default_na=False)
    review = pd.read_csv(REVIEW, keep_default_na=False)
    quarantine = set(review.loc[review["decision"] == "quarantine", "external_image_id"].astype(int))
    if quarantine != {2714443, 2688988}:
        raise RuntimeError(f"Unexpected frozen review quarantine set: {sorted(quarantine)}")
    source = json.loads(TRAIN_METADATA.read_text(encoding="utf-8"))
    categories = {int(row["id"]): row for row in source["categories"]}
    images = {int(row["id"]): row for row in source["images"]}
    # The base manifest already records the exact ROBird taxon-to-iNaturalist
    # mapping. Only the two quarantined rows are replaced from train_mini.
    replacement_taxa = {
        int(row.taxon_id): str(row.external_scientific_name)
        for row in base.itertuples(index=False)
        if int(row.external_image_id) in quarantine
    }
    robird_hashes = set(robird["sha256"].astype(str))
    robird_dhashes = [dhash(Path(row.local_path)) for row in robird.itertuples(index=False)]
    replacement_rows: list[dict[str, Any]] = []
    for taxon_id, external_name in sorted(replacement_taxa.items()):
        candidates: list[dict[str, Any]] = []
        for annotation in source["annotations"]:
            category = categories[int(annotation["category_id"])]
            if category.get("class") != "Aves" or str(category["name"]) != external_name:
                continue
            image = images[int(annotation["image_id"])]
            local_path = TRAIN_IMAGE_ROOT / str(image["file_name"]).replace("train_mini/", "")
            if not local_path.exists():
                continue
            with Image.open(local_path) as opened:
                width, height = opened.size
                opened.verify()
            digest = sha256_file(local_path)
            if digest in robird_hashes:
                continue
            candidate_hash = dhash(local_path)
            if any((candidate_hash ^ other).bit_count() <= 4 for other in robird_dhashes):
                continue
            candidates.append(
                {
                    "external_image_id": int(image["id"]),
                    "external_category_id": int(category["id"]),
                    "external_scientific_name": external_name,
                    "external_common_name": str(category.get("common_name", "")),
                    "rights_holder": str(image.get("rights_holder", "")),
                    "date": str(image.get("date", "")),
                    "latitude": image.get("latitude"),
                    "longitude": image.get("longitude"),
                    "license_id": int(image.get("license", -1)),
                    "local_path": str(local_path),
                    "width": int(width),
                    "height": int(height),
                    "sha256": digest,
                }
            )
        candidates.sort(key=lambda row: row["external_image_id"])
        if not candidates:
            raise RuntimeError(f"No clean replacement candidate for taxon {taxon_id} / {external_name}")
        selected = candidates[0]
        base_row = base.loc[base["taxon_id"].astype(int) == taxon_id].iloc[0]
        selected.update(
            {
                "external_rank_within_taxon": 10,
                "source_dataset": "iNaturalist-2021-train_mini-replacement",
                "taxon_id": int(base_row["taxon_id"]),
                "class_index": int(base_row["class_index"]),
                "scientific_name": str(base_row["scientific_name"]),
                "common_name": str(base_row["common_name"]),
                "group_id": int(selected["external_image_id"]),
                "split": "g6x_external_test",
                "observer_identity_status": "rights_holder_only_cross_source",
            }
        )
        replacement_rows.append(selected)

    retained = base.loc[~base["external_image_id"].astype(int).isin(quarantine)].copy()
    replacement = pd.DataFrame(replacement_rows)
    manifest = pd.concat([retained, replacement], ignore_index=True, sort=False)
    manifest = manifest.sort_values(["class_index", "external_image_id"]).reset_index(drop=True)
    manifest["external_row_id"] = range(len(manifest))
    ordered = [
        "external_row_id", "external_image_id", "external_category_id", "external_rank_within_taxon",
        "source_dataset", "taxon_id", "class_index", "scientific_name", "common_name",
        "external_scientific_name", "external_common_name", "rights_holder", "date", "latitude",
        "longitude", "license_id", "local_path", "width", "height", "sha256", "group_id",
        "split", "observer_identity_status",
    ]
    manifest = manifest[ordered]
    counts = manifest.groupby("taxon_id").size()
    if len(manifest) != 1000 or len(counts) != 100 or int(counts.min()) != 10 or int(counts.max()) != 10:
        raise RuntimeError("G6X-v1.1 does not have exactly 1,000 images and ten per taxon")
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_MANIFEST, index=False)
    mapping = manifest[["taxon_id", "class_index", "scientific_name", "common_name", "external_scientific_name", "source_dataset"]].drop_duplicates("taxon_id").sort_values("class_index")
    mapping.to_csv(OUT_MAPPING, index=False)
    audit = {
        "status": "PASS_G6X_EXTERNAL_METADATA_V1_1",
        "gate_pass": True,
        "claim_scope": "CROSS_SOURCE_SINGLE_PHOTO_TRANSFER_ONLY",
        "original_p0_g6_status": "UNREGISTERED_FALSE",
        "counts": {
            "images": int(len(manifest)),
            "taxa": int(manifest["taxon_id"].nunique()),
            "images_per_taxon_min": int(counts.min()),
            "images_per_taxon_max": int(counts.max()),
            "source_dataset_counts": {str(key): int(value) for key, value in manifest["source_dataset"].value_counts().items()},
        },
        "quarantined_v1_external_image_ids": sorted(quarantine),
        "replacement_rows": replacement_rows,
        "provenance": {
            "base_manifest_sha256": sha256_file(BASE_MANIFEST),
            "review_sha256": sha256_file(REVIEW),
            "manifest_sha256": sha256_file(OUT_MANIFEST),
            "taxonomy_mapping_sha256": sha256_file(OUT_MAPPING),
        },
        "review_note": "Two v1 rows were quarantined after Codex-assisted visual inspection of dHash candidates and replaced by clean train_mini images; no G6X test tuning was performed.",
    }
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
