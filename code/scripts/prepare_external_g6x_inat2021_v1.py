from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBIRD_MANIFEST = PROJECT_ROOT / "code" / "data" / "manifests" / "development_photos_v5_3.csv"
INAT_ROOT = Path("E:/Datasets/iNaturalist-2021")
INAT_METADATA = INAT_ROOT / "metadata" / "val.json"
INAT_IMAGE_ROOT = INAT_ROOT / "bird_images" / "test"
OUT_MANIFEST = PROJECT_ROOT / "code" / "data" / "manifests" / "external_g6x_inat2021_v1.csv"
OUT_AUDIT = PROJECT_ROOT / "code" / "results" / "external_g6x_inat2021_v1_audit.json"
OUT_MAPPING = PROJECT_ROOT / "code" / "results" / "external_g6x_inat2021_v1_taxonomy_mapping.csv"

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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frozen G6X external iNaturalist-2021 manifest.")
    parser.add_argument("--force", action="store_true", help="Allow writing a new version only if outputs do not exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if OUT_MANIFEST.exists() or OUT_AUDIT.exists() or OUT_MAPPING.exists():
        raise FileExistsError("G6X-v1 outputs already exist; use a new version rather than overwriting")
    if not ROBIRD_MANIFEST.exists() or not INAT_METADATA.exists() or not INAT_IMAGE_ROOT.exists():
        raise FileNotFoundError("ROBIRD manifest or iNaturalist-2021 extracted validation images are missing")

    robird = pd.read_csv(ROBIRD_MANIFEST, keep_default_na=False)
    taxa = robird[["taxon_id", "class_index", "scientific_name", "common_name"]].drop_duplicates("taxon_id").copy()
    if len(taxa) != 100:
        raise RuntimeError(f"Expected 100 ROBird taxa, found {len(taxa)}")
    taxa["scientific_name"] = taxa["scientific_name"].astype(str).str.strip()
    taxa["common_name"] = taxa["common_name"].astype(str).str.strip()
    target_by_external_name: dict[str, dict[str, Any]] = {}
    mapping_rows: list[dict[str, Any]] = []
    for row in taxa.to_dict("records"):
        canonical = row["scientific_name"]
        if not canonical or canonical.lower() == "nan":
            if row["common_name"].lower() != "pacific loon":
                raise RuntimeError(f"Missing scientific name without frozen alias: {row}")
            external_name = "Gavia pacifica"
            mapping_type = "common_name_rescue_missing_robird_scientific_name"
        else:
            external_name = ALIASES.get(canonical, canonical)
            mapping_type = "scientific_name_exact" if external_name == canonical else "taxonomic_synonym"
        if external_name in target_by_external_name:
            raise RuntimeError(f"External taxonomy name maps to multiple ROBird taxa: {external_name}")
        target_by_external_name[external_name] = row
        mapping_rows.append(
            {
                "taxon_id": int(row["taxon_id"]),
                "class_index": int(row["class_index"]),
                "robird_scientific_name": canonical,
                "robird_common_name": row["common_name"],
                "inat_scientific_name": external_name,
                "mapping_type": mapping_type,
            }
        )

    source = load_json(INAT_METADATA)
    categories = {int(row["id"]): row for row in source["categories"]}
    images = {int(row["id"]): row for row in source["images"]}
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in target_by_external_name}
    for annotation in source["annotations"]:
        category = categories[int(annotation["category_id"])]
        external_name = str(category["name"])
        if category.get("class") != "Aves" or external_name not in target_by_external_name:
            continue
        image = images[int(annotation["image_id"])]
        local_rel = str(image["file_name"]).replace("val/", "")
        local_path = INAT_IMAGE_ROOT / local_rel
        candidates[external_name].append(
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
            }
        )

    selected: list[dict[str, Any]] = []
    for external_name, robird_row in target_by_external_name.items():
        rows = sorted(candidates[external_name], key=lambda value: value["external_image_id"])
        if len(rows) != 10:
            raise RuntimeError(f"Expected exactly 10 iNaturalist-2021 validation images for {external_name}, found {len(rows)}")
        for rank, row in enumerate(rows, start=1):
            path = Path(row["local_path"])
            if not path.exists():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                width, height = image.size
                image.verify()
            row = dict(row)
            row.update(
                {
                    "external_rank_within_taxon": rank,
                    "source_dataset": "iNaturalist-2021-val",
                    "taxon_id": int(robird_row["taxon_id"]),
                    "class_index": int(robird_row["class_index"]),
                    "scientific_name": str(robird_row["scientific_name"]),
                    "common_name": str(robird_row["common_name"]),
                    "width": int(width),
                    "height": int(height),
                    "sha256": sha256_file(path),
                }
            )
            selected.append(row)

    manifest = pd.DataFrame(selected).sort_values(["class_index", "external_image_id"]).reset_index(drop=True)
    manifest.insert(0, "external_row_id", range(len(manifest)))
    manifest["group_id"] = manifest["external_image_id"].astype(int)
    manifest["split"] = "g6x_external_test"
    manifest["observer_identity_status"] = "rights_holder_only_cross_source"
    manifest["local_path"] = manifest["local_path"].astype(str)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT_MANIFEST, index=False)
    pd.DataFrame(mapping_rows).sort_values("class_index").to_csv(OUT_MAPPING, index=False)

    robird_hashes = set(robird["sha256"].astype(str)) if "sha256" in robird else set()
    exact_overlap = sorted(set(manifest["sha256"]) & robird_hashes)
    counts = manifest.groupby("taxon_id").size()
    audit = {
        "status": "PASS_G6X_EXTERNAL_METADATA" if not exact_overlap else "STOP_G6X_EXACT_OVERLAP",
        "gate_pass": not exact_overlap and len(manifest) == 1000 and len(counts) == 100 and int(counts.min()) == 10 and int(counts.max()) == 10,
        "claim_scope": "CROSS_SOURCE_SINGLE_PHOTO_TRANSFER_ONLY",
        "original_p0_g6_status": "UNREGISTERED_FALSE",
        "source": {
            "metadata": str(INAT_METADATA),
            "metadata_sha256": sha256_file(INAT_METADATA),
            "image_root": str(INAT_IMAGE_ROOT),
            "robird_manifest": str(ROBIRD_MANIFEST),
            "robird_manifest_sha256": sha256_file(ROBIRD_MANIFEST),
        },
        "counts": {
            "images": int(len(manifest)),
            "taxa": int(manifest["taxon_id"].nunique()),
            "images_per_taxon_min": int(counts.min()),
            "images_per_taxon_max": int(counts.max()),
            "licenses": {str(key): int(value) for key, value in manifest["license_id"].value_counts().sort_index().items()},
        },
        "exact_sha256_overlap_with_robird": exact_overlap,
        "observer_identity": "not_comparable_to_ROBird_numeric_observer_ids; rights_holder retained for stratification",
        "provenance": {
            "manifest_sha256": sha256_file(OUT_MANIFEST),
            "taxonomy_mapping_sha256": sha256_file(OUT_MAPPING),
        },
    }
    OUT_AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if audit["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
