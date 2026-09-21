from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from robird.io import sha256_file
from robird.manifest import validate_manifest_schema


def difference_hash(path: Path, hash_size: int = 8) -> int:
    if hash_size < 1:
        raise ValueError("hash_size must be positive")
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(grayscale, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(left: int, right: int) -> int:
    return int((int(left) ^ int(right)).bit_count())


def duplicate_candidates(
    frame: pd.DataFrame, hash_size: int = 8, threshold: int = 4
) -> pd.DataFrame:
    if threshold < 0 or threshold >= hash_size * hash_size:
        raise ValueError("threshold must be in [0, hash_size ** 2)")
    hashes: list[tuple[int, int, int, int]] = []
    for row in frame.itertuples(index=False):
        path = Path(str(row.local_path))
        if not path.is_file():
            continue
        hashes.append(
            (int(row.photo_id), int(row.observation_id), int(row.observer_id), difference_hash(path, hash_size))
        )
    bit_count = hash_size * hash_size
    block_count = threshold + 1
    block_widths = [bit_count // block_count] * block_count
    for index in range(bit_count % block_count):
        block_widths[index] += 1
    shifts: list[tuple[int, int]] = []
    remaining = bit_count
    for width in block_widths:
        remaining -= width
        shifts.append((remaining, (1 << width) - 1))

    buckets: dict[tuple[int, int], list[int]] = {}
    candidate_indices: set[tuple[int, int]] = set()
    for position, record in enumerate(hashes):
        for block_index, (shift, mask) in enumerate(shifts):
            key = (block_index, (record[3] >> shift) & mask)
            for previous in buckets.get(key, []):
                candidate_indices.add((previous, position))
            buckets.setdefault(key, []).append(position)

    candidates: list[dict[str, Any]] = []
    for left_pos, right_pos in sorted(candidate_indices):
        left, right = hashes[left_pos], hashes[right_pos]
        if left[1] == right[1]:
            continue
        distance = hamming_distance(left[3], right[3])
        if distance <= threshold:
            candidates.append(
                {
                    "left_photo_id": left[0],
                    "right_photo_id": right[0],
                    "left_observation_id": left[1],
                    "right_observation_id": right[1],
                    "same_observer": left[2] == right[2],
                    "hamming_distance": distance,
                }
            )
    columns = [
        "left_photo_id",
        "right_photo_id",
        "left_observation_id",
        "right_observation_id",
        "same_observer",
        "hamming_distance",
    ]
    return pd.DataFrame.from_records(candidates, columns=columns)


def _observer_concentration(groups: pd.DataFrame) -> pd.DataFrame:
    counts = (
        groups.groupby(["taxon_id", "observer_id"])["observation_id"]
        .nunique()
        .rename("observer_groups")
        .reset_index()
    )
    totals = groups.groupby("taxon_id")["observation_id"].nunique().rename("species_groups")
    counts = counts.join(totals, on="taxon_id")
    counts["fraction"] = counts["observer_groups"] / counts["species_groups"]
    return counts


def audit_manifest(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    inspect_files: bool = False,
    near_duplicate_audit_passed: bool = False,
    near_duplicate_status: str = "not_audited",
    holdout_registered: bool = False,
    holdout_status: str = "not_registered",
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Cannot audit an empty manifest")
    schema_violations = validate_manifest_schema(frame)
    allowed = {str(value).lower() for value in config["licenses"]["allowed"]}
    license_codes = frame["license_code"].fillna("").astype(str).str.lower()
    invalid_license_rows = frame.loc[~license_codes.isin(allowed)]
    missing_attribution = frame["attribution"].fillna("").astype(str).str.strip().eq("")
    missing_url = frame["url"].fillna("").astype(str).str.strip().eq("")
    local_paths = frame["local_path"].fillna("").astype(str).str.strip()
    missing_local_path = local_paths.eq("") | ~local_paths.map(lambda value: Path(value).is_absolute())
    sha_values = frame["sha256"].fillna("").astype(str).str.strip().str.lower()
    valid_sha = sha_values.str.fullmatch(r"[0-9a-f]{64}", na=False)
    missing_files: list[str] = []
    hash_mismatches: list[dict[str, Any]] = []
    if inspect_files:
        for row in frame.itertuples(index=False):
            path = Path(str(row.local_path))
            if not path.is_file():
                missing_files.append(str(path))
                continue
            expected = str(row.sha256).strip().lower()
            actual = sha256_file(path)
            if expected != actual:
                hash_mismatches.append(
                    {"photo_id": int(row.photo_id), "expected": expected, "actual": actual}
                )

    groups = frame.drop_duplicates("observation_id")
    group_sizes = frame.groupby("observation_id")["photo_id"].nunique()
    per_species_groups = groups.groupby("taxon_id")["observation_id"].nunique()
    per_species_observers = groups.groupby("taxon_id")["observer_id"].nunique()
    concentration = _observer_concentration(groups)
    exact_hashes = frame.loc[frame["sha256"].fillna("").astype(str).str.len() > 0]
    exact_duplicate_hashes = exact_hashes.loc[
        exact_hashes["sha256"].duplicated(keep=False), "sha256"
    ].nunique()

    eligibility = config["eligibility"]
    group_size_pass = bool(
        group_sizes.between(
            int(eligibility["min_photos_per_group"]),
            int(eligibility["max_photos_per_group"]),
        ).all()
    )
    max_fraction = float(concentration["fraction"].max()) if len(concentration) else 1.0
    gates = {
        "P0_G0_PROVENANCE": not schema_violations
        and len(invalid_license_rows) == 0
        and not bool(missing_attribution.any())
        and not bool(missing_url.any())
        and not bool(missing_local_path.any())
        and bool(valid_sha.all())
        and bool(inspect_files)
        and len(missing_files) == 0
        and len(hash_mismatches) == 0,
        "P0_G1_SCALE": frame["taxon_id"].nunique() >= int(eligibility["min_species"])
        and groups["observation_id"].nunique() >= int(eligibility["min_total_groups"])
        and len(frame) >= int(eligibility["min_total_photos"]),
        "P0_G2_SPECIES_SUPPORT": int(per_species_groups.min())
        >= int(eligibility["min_groups_per_species"])
        and int(per_species_observers.min()) >= int(eligibility["min_observers_per_species"])
        and max_fraction <= float(eligibility["max_observer_fraction_per_species"]),
        "P0_G3_GROUP_INTEGRITY": group_size_pass and not schema_violations,
        "P0_G4_EXACT_DUPLICATES": int(exact_duplicate_hashes) == 0,
        "P0_G5_NEAR_DUPLICATES": bool(near_duplicate_audit_passed),
        "P0_G6_UNSEEN_HOLDOUT": holdout_registered,
    }
    status = str(config["protocol"]["status"]).lower()
    development_gates = {key: value for key, value in gates.items() if key != "P0_G6_UNSEEN_HOLDOUT"}
    if status == "frozen" and not all(development_gates.values()):
        decision = "STOP_DATASET_P0"
    elif status == "frozen" and gates["P0_G6_UNSEEN_HOLDOUT"]:
        decision = "CONTINUE_TO_COMMON_BENCHMARK"
    elif status == "frozen":
        decision = "CONTINUE_DEVELOPMENT_ONLY"
    else:
        decision = "DRAFT_ONLY_NO_SCIENTIFIC_DECISION"
    return {
        "protocol": dict(config["protocol"]),
        "decision": decision,
        "counts": {
            "photos": int(len(frame)),
            "groups": int(groups["observation_id"].nunique()),
            "observers": int(groups["observer_id"].nunique()),
            "species": int(groups["taxon_id"].nunique()),
            "cohorts": {str(k): int(v) for k, v in groups["cohort"].value_counts().items()},
        },
        "support": {
            "min_groups_per_species": int(per_species_groups.min()),
            "min_observers_per_species": int(per_species_observers.min()),
            "max_observer_fraction_per_species": max_fraction,
            "min_group_size": int(group_sizes.min()),
            "max_group_size": int(group_sizes.max()),
        },
        "violations": {
            "schema": schema_violations,
            "invalid_license_rows": int(len(invalid_license_rows)),
            "missing_attribution_rows": int(missing_attribution.sum()),
            "missing_url_rows": int(missing_url.sum()),
            "missing_or_relative_local_path_rows": int(missing_local_path.sum()),
            "invalid_sha256_rows": int((~valid_sha).sum()),
            "missing_files": missing_files,
            "hash_mismatches": hash_mismatches,
            "exact_duplicate_hashes": int(exact_duplicate_hashes),
            "near_duplicate_status": str(near_duplicate_status),
            "holdout_status": str(holdout_status),
        },
        "gates": gates,
        "development_gates": development_gates,
    }
