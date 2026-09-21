from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "row_id",
    "cohort",
    "observation_id",
    "observer_id",
    "taxon_id",
    "class_index",
    "scientific_name",
    "common_name",
    "observed_on",
    "created_at",
    "place_ids",
    "photo_id",
    "license_code",
    "attribution",
    "url",
    "local_path",
    "sha256",
    "width",
    "height",
    "legacy_split",
]


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _source_photo_metadata(path: Path) -> dict[int, dict[str, Any]]:
    source = _load_json(path)
    metadata: dict[int, dict[str, Any]] = {}
    for group in source.get("groups", []):
        group_fields = {
            "observation_id": group.get("observation_id"),
            "observer_id": group.get("observer_id"),
            "taxon_id": group.get("taxon_id"),
            "scientific_name": group.get("scientific_name", ""),
            "common_name": group.get("species_name", group.get("crosswalk_name", "")),
            "observed_on": group.get("observed_on", ""),
            "created_at": group.get("created_at", ""),
            "place_ids": json.dumps(group.get("place_ids", []), separators=(",", ":")),
        }
        for photo in group.get("photos", []):
            photo_id = int(photo["photo_id"])
            if photo_id in metadata:
                raise ValueError(f"Duplicate photo_id in source manifest: {photo_id}")
            metadata[photo_id] = {
                **group_fields,
                "photo_id": photo_id,
                "license_code": photo.get("license_code", ""),
                "attribution": photo.get("attribution", ""),
                "url": photo.get("url", ""),
                "width": photo.get("original_width", ""),
                "height": photo.get("original_height", ""),
            }
    return metadata


def _download_metadata(path: Path) -> dict[int, dict[str, Any]]:
    ledger_path = Path(path).resolve()
    ledger = _load_json(ledger_path)
    rows: dict[int, dict[str, Any]] = {}
    for photo in ledger.get("photos", []):
        photo_id = int(photo["photo_id"])
        if photo_id in rows:
            raise ValueError(f"Duplicate photo_id in download ledger: {photo_id}")
        local_value = str(photo.get("path", "")).strip()
        local_path = Path(local_value)
        if local_value and not local_path.is_absolute():
            local_path = (ledger_path.parent / local_path).resolve()
        rows[photo_id] = {
            "local_path": str(local_path) if local_value else "",
            "sha256": photo.get("sha256", ""),
            "download_ok": bool(photo.get("ok", False)),
        }
    return rows


def normalize_legacy_manifest(
    csv_path: Path,
    source_json: Path,
    download_ledger: Path,
    cohort: str,
) -> pd.DataFrame:
    legacy = pd.read_csv(csv_path)
    required = {"observation_id", "observer_id", "taxon_id", "photo_id", "license_code"}
    missing = sorted(required - set(legacy.columns))
    if missing:
        raise ValueError(f"Legacy manifest missing columns {missing}: {csv_path}")

    source = _source_photo_metadata(source_json)
    downloads = _download_metadata(download_ledger)
    records: list[dict[str, Any]] = []
    for row in legacy.to_dict(orient="records"):
        photo_id = int(row["photo_id"])
        if photo_id not in source:
            raise ValueError(f"Photo {photo_id} missing from source metadata {source_json}")
        if photo_id not in downloads:
            raise ValueError(f"Photo {photo_id} missing from download ledger {download_ledger}")
        source_row = source[photo_id]
        download_row = downloads[photo_id]
        for field in ("observation_id", "observer_id", "taxon_id"):
            source_value = source_row.get(field)
            if source_value is not None and int(source_value) != int(row[field]):
                raise ValueError(
                    f"Photo {photo_id} has conflicting {field}: "
                    f"legacy={row[field]}, source={source_value}"
                )
        legacy_license = str(row.get("license_code", "")).strip().lower()
        source_license = str(source_row.get("license_code", "")).strip().lower()
        if legacy_license and source_license and legacy_license != source_license:
            raise ValueError(
                f"Photo {photo_id} has conflicting license codes: "
                f"legacy={legacy_license}, source={source_license}"
            )
        if not download_row["download_ok"]:
            raise ValueError(f"Photo {photo_id} is not marked downloaded")
        records.append(
            {
                "cohort": cohort,
                "observation_id": int(row["observation_id"]),
                "observer_id": int(row["observer_id"]),
                "taxon_id": int(row["taxon_id"]),
                "scientific_name": str(row.get("scientific_name", source_row["scientific_name"])),
                "common_name": str(row.get("species_name", source_row["common_name"])),
                "observed_on": str(row.get("observed_on", source_row["observed_on"])),
                "created_at": str(row.get("created_at", source_row["created_at"])),
                "place_ids": source_row["place_ids"],
                "photo_id": photo_id,
                "license_code": str(source_row["license_code"]).lower(),
                "attribution": str(source_row["attribution"]),
                "url": str(source_row["url"]),
                "local_path": str(download_row["local_path"]),
                "sha256": str(download_row["sha256"]),
                "width": source_row["width"],
                "height": source_row["height"],
                "legacy_split": str(row.get("split", "")),
            }
        )
    return pd.DataFrame.from_records(records)


def combine_seed_manifests(original: pd.DataFrame, confirmation: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([original, confirmation], ignore_index=True, sort=False)
    if frame.empty:
        raise ValueError("Cannot combine empty seed manifests")
    duplicate_photos = frame.loc[frame["photo_id"].duplicated(keep=False), "photo_id"].unique()
    if len(duplicate_photos):
        raise ValueError(f"Photo IDs overlap across cohorts: {duplicate_photos[:10].tolist()}")
    taxa = sorted(int(value) for value in frame["taxon_id"].unique())
    class_map = {taxon_id: index for index, taxon_id in enumerate(taxa)}
    frame["class_index"] = frame["taxon_id"].map(class_map).astype(np.int64)
    frame = frame.sort_values(["taxon_id", "observation_id", "photo_id"]).reset_index(drop=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    for column in CANONICAL_COLUMNS:
        if column not in frame:
            frame[column] = ""
    return frame[CANONICAL_COLUMNS]


def validate_manifest_schema(frame: pd.DataFrame) -> list[str]:
    violations: list[str] = []
    missing = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
    if missing:
        return [f"missing columns: {missing}"]
    for column in ["observation_id", "observer_id", "taxon_id", "photo_id", "cohort"]:
        if frame[column].isna().any() or (frame[column].astype(str).str.len() == 0).any():
            violations.append(f"missing values in {column}")
    if frame["photo_id"].duplicated().any():
        violations.append("duplicate photo_id")
    for observation_id, group in frame.groupby("observation_id", sort=False):
        if group["observer_id"].nunique() != 1:
            violations.append(f"observation {observation_id} has multiple observers")
        if group["taxon_id"].nunique() != 1:
            violations.append(f"observation {observation_id} has multiple taxa")
        if group["cohort"].nunique() != 1:
            violations.append(f"observation {observation_id} spans cohorts")
    return violations
