from __future__ import annotations

import hashlib
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .io import sha256_file
from .manifest import CANONICAL_COLUMNS


USER_AGENT = "ROBird-Bench metadata feasibility audit (academic research)"


def stable_hash_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_seed_identity(
    original_csv: Path, confirmation_csv: Path
) -> tuple[list[dict[str, Any]], set[int], set[int], dict[str, Any]]:
    frames = [pd.read_csv(original_csv), pd.read_csv(confirmation_csv)]
    required = {"observation_id", "observer_id", "taxon_id", "species_name"}
    for path, frame in zip((original_csv, confirmation_csv), frames):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Seed manifest {path} is missing columns: {missing}")
    if "scientific_name" not in frames[0].columns:
        raise ValueError(f"Original seed manifest {original_csv} is missing scientific_name")

    taxon_sets = [set(frame["taxon_id"].astype(np.int64)) for frame in frames]
    if taxon_sets[0] != taxon_sets[1]:
        raise ValueError("The two inspected cohorts do not contain the same taxon set")
    observation_sets = [set(frame["observation_id"].astype(np.int64)) for frame in frames]
    observer_sets = [set(frame["observer_id"].astype(np.int64)) for frame in frames]
    if observation_sets[0] & observation_sets[1]:
        raise ValueError("The two inspected cohorts overlap in observation_id")
    if observer_sets[0] & observer_sets[1]:
        raise ValueError("The two inspected cohorts overlap in observer_id")

    combined = pd.concat(frames, ignore_index=True)
    group_counts = combined.groupby("taxon_id")["observation_id"].nunique()
    if group_counts.nunique() != 1 or int(group_counts.iloc[0]) != 26:
        raise ValueError(f"Expected exactly 26 seed groups per taxon, got {group_counts.to_dict()}")

    scientific_by_taxon: dict[int, str] = {}
    for taxon_id, rows in frames[0].groupby("taxon_id", sort=True):
        scientific = sorted(set(rows["scientific_name"].astype(str).str.strip()))
        if len(scientific) != 1:
            raise ValueError(f"Inconsistent original scientific names for taxon_id={taxon_id}")
        scientific_by_taxon[int(taxon_id)] = scientific[0]

    taxa: list[dict[str, Any]] = []
    for taxon_id, rows in combined.groupby("taxon_id", sort=True):
        common = sorted(set(rows["species_name"].astype(str).str.strip()))
        if len(common) != 1:
            raise ValueError(f"Inconsistent seed names for taxon_id={taxon_id}")
        scientific_name = scientific_by_taxon[int(taxon_id)]
        taxa.append(
            {
                "taxon_id": int(taxon_id),
                "scientific_name": scientific_name,
                "common_name": common[0],
                "genus": scientific_name.split()[0],
            }
        )

    all_observations = observation_sets[0] | observation_sets[1]
    all_observers = observer_sets[0] | observer_sets[1]
    audit = {
        "species": len(taxa),
        "observations": len(all_observations),
        "observers": len(all_observers),
        "groups_per_species": int(group_counts.iloc[0]),
        "observation_overlap": 0,
        "observer_overlap": 0,
        "original_manifest_sha256": sha256_file(original_csv),
        "confirmation_manifest_sha256": sha256_file(confirmation_csv),
    }
    return taxa, all_observations, all_observers, audit


def load_inat2021_birds(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.strip().lower():
        raise ValueError(f"iNaturalist 2021 taxonomy hash mismatch: expected {expected_sha256}, got {actual}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = payload.get("categories", [])
    birds = [row for row in categories if row.get("class") == "Aves"]
    if not birds:
        raise ValueError("No Aves categories found in the frozen iNaturalist 2021 metadata")
    ids = [int(row["id"]) for row in birds]
    names = [str(row["name"]).strip().casefold() for row in birds]
    if len(ids) != len(set(ids)) or len(names) != len(set(names)):
        raise ValueError("Frozen Aves taxonomy contains duplicate category IDs or scientific names")
    return birds


def rank_expansion_taxa(
    categories: Iterable[Mapping[str, Any]], current_taxa: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    categories = list(categories)
    current_taxa = list(current_taxa)
    by_name = {str(row["name"]).strip().casefold(): row for row in categories}
    current_names = {str(row["scientific_name"]).strip().casefold() for row in current_taxa}
    current_genera = {str(row["scientific_name"]).split()[0].casefold() for row in current_taxa}
    current_families = {
        str(by_name[name].get("family", "")).strip().casefold()
        for name in current_names
        if name in by_name and str(by_name[name].get("family", "")).strip()
    }

    ranked: list[dict[str, Any]] = []
    for row in categories:
        scientific_name = str(row["name"]).strip()
        if scientific_name.casefold() in current_names:
            continue
        genus = str(row.get("genus", scientific_name.split()[0])).strip()
        family = str(row.get("family", "")).strip()
        if genus.casefold() in current_genera:
            priority = 0
            reason = "same_genus"
        elif family and family.casefold() in current_families:
            priority = 1
            reason = "same_family"
        else:
            priority = 2
            reason = "other_aves"
        ranked.append(
            {
                "category_id": int(row["id"]),
                "scientific_name": scientific_name,
                "common_name": str(row.get("common_name", "")).strip(),
                "family": family,
                "genus": genus,
                "priority": priority,
                "priority_reason": reason,
            }
        )
    return sorted(ranked, key=lambda row: (row["priority"], row["category_id"]))


def api_get_json(url: str, attempts: int) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response:
                value = json.load(response)
            if not isinstance(value, dict):
                raise ValueError(f"API returned a non-object payload for {url}")
            return value
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
        ) as error:
            errors.append(f"{type(error).__name__}: {error}")
            if attempt + 1 == attempts:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"iNaturalist request failed after {attempts} attempts: {errors[-1]}")


def _api_url(base: str, endpoint: str, params: Mapping[str, Any]) -> str:
    return f"{base.rstrip('/')}/{endpoint.lstrip('/')}?{urllib.parse.urlencode(params)}"


def resolve_exact_species(
    api_base: str, scientific_name: str, *, attempts: int = 5
) -> dict[str, Any] | None:
    payload = api_get_json(
        _api_url(
            api_base,
            "taxa/autocomplete",
            {"q": scientific_name, "rank": "species", "per_page": 30},
        ),
        attempts,
    )
    wanted = scientific_name.strip().casefold()
    matches = [
        row
        for row in payload.get("results", [])
        if str(row.get("name", "")).strip().casefold() == wanted
        and row.get("rank") == "species"
        and row.get("is_active", True)
        and row.get("iconic_taxon_name") == "Aves"
    ]
    if len(matches) != 1:
        return None
    row = matches[0]
    return {
        "taxon_id": int(row["id"]),
        "scientific_name": str(row["name"]),
        "common_name": str(row.get("preferred_common_name", "")),
    }


def _large_url(url: str) -> str:
    path, separator, query = url.partition("?")
    filename = path.rsplit("/", 1)[-1]
    stem, dot, extension = filename.rpartition(".")
    if stem in {"square", "small", "medium", "large"} and dot and extension:
        path = path[: -len(filename)] + f"large.{extension}"
    else:
        raise ValueError(f"Unexpected iNaturalist photo URL: {url}")
    return path + (separator + query if separator else "")


def fetch_observation_candidates(
    taxon: Mapping[str, Any],
    source: Mapping[str, Any],
    excluded_observations: set[int],
    excluded_observers: set[int],
    *,
    seed: int,
    early_stop_unique_observers: int | None = None,
) -> list[dict[str, Any]]:
    allowed = {str(value).strip().casefold() for value in source["photo_licenses"]}
    by_observation: dict[int, dict[str, Any]] = {}
    for page in range(1, int(source["pages"]) + 1):
        params = {
            "taxon_id": int(taxon["taxon_id"]),
            "photos": "true",
            "photo_license": ",".join(source["photo_licenses"]),
            "quality_grade": source["quality_grade"],
            "created_d2": source["created_d2"],
            "per_page": int(source["per_page"]),
            "page": page,
            "order": "desc",
            "order_by": "created_at",
        }
        payload = api_get_json(
            _api_url(str(source["api_base"]), "observations", params),
            int(source["request_attempts"]),
        )
        for observation in payload.get("results", []):
            if not observation.get("user"):
                continue
            observation_id = int(observation["id"])
            observer_id = int(observation["user"]["id"])
            if observation_id in excluded_observations or observer_id in excluded_observers:
                continue
            observed_taxon = observation.get("taxon") or {}
            lineage = {int(value) for value in observed_taxon.get("ancestor_ids", [])}
            if observed_taxon.get("id") is not None:
                lineage.add(int(observed_taxon["id"]))
            if int(taxon["taxon_id"]) not in lineage:
                continue
            photos = []
            for photo in observation.get("photos", []):
                license_code = str(photo.get("license_code", "")).strip().casefold()
                url = str(photo.get("url", "")).strip()
                if license_code not in allowed or photo.get("hidden", False) or not url:
                    continue
                dimensions = photo.get("original_dimensions") or {}
                photos.append(
                    {
                        "photo_id": int(photo["id"]),
                        "license_code": license_code,
                        "attribution": str(photo.get("attribution", "")),
                        "width": dimensions.get("width"),
                        "height": dimensions.get("height"),
                        "url": _large_url(url),
                    }
                )
            if len(photos) < int(source["min_photos_per_group"]):
                continue
            photos = sorted(
                photos,
                key=lambda photo: stable_hash_int(
                    seed,
                    "scaleup-photo",
                    taxon["taxon_id"],
                    observation_id,
                    photo["photo_id"],
                ),
            )[: int(source["max_photos_per_group"])]
            by_observation[observation_id] = {
                "taxon_id": int(taxon["taxon_id"]),
                "scientific_name": str(taxon["scientific_name"]),
                "common_name": str(taxon.get("common_name", "")),
                "observation_id": observation_id,
                "observer_id": observer_id,
                "observed_on": observation.get("observed_on"),
                "created_at": observation.get("created_at"),
                "place_ids": observation.get("place_ids", []),
                "photos": photos,
            }
        if early_stop_unique_observers is not None:
            unique_observers = {row["observer_id"] for row in by_observation.values()}
            if len(unique_observers) >= early_stop_unique_observers:
                break
        time.sleep(float(source["request_delay_seconds"]))
    return list(by_observation.values())


def compact_candidates_by_observer(
    candidates: Iterable[Mapping[str, Any]], cap: int, seed: int, taxon_id: int
) -> list[dict[str, Any]]:
    by_observer: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        observer_id = int(candidate["observer_id"])
        row = dict(candidate)
        incumbent = by_observer.get(observer_id)
        key = stable_hash_int(seed, "observer-choice", taxon_id, observer_id, row["observation_id"])
        incumbent_key = (
            stable_hash_int(seed, "observer-choice", taxon_id, observer_id, incumbent["observation_id"])
            if incumbent is not None
            else None
        )
        if incumbent is None or key < int(incumbent_key):
            by_observer[observer_id] = row
    rows = sorted(
        by_observer.values(),
        key=lambda row: stable_hash_int(seed, "candidate-cap", taxon_id, row["observation_id"]),
    )
    return rows[:cap]


def select_balanced_groups(
    candidates: Iterable[Mapping[str, Any]],
    targets: Mapping[int, int],
    seed: int,
    *,
    observer_capacity: int = 1,
    time_limit_seconds: float = 300.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [dict(row) for row in candidates if int(row["taxon_id"]) in targets]
    species = sorted(int(value) for value in targets)
    observers = sorted({int(row["observer_id"]) for row in rows})
    species_index = {value: index for index, value in enumerate(species)}
    observer_index = {value: index for index, value in enumerate(observers)}

    independent_counts = {
        str(taxon_id): len({int(row["observer_id"]) for row in rows if int(row["taxon_id"]) == taxon_id})
        for taxon_id in species
    }
    insufficient = {
        str(taxon_id): {"available": independent_counts[str(taxon_id)], "target": int(targets[taxon_id])}
        for taxon_id in species
        if independent_counts[str(taxon_id)] < int(targets[taxon_id])
    }
    base_audit: dict[str, Any] = {
        "candidate_groups": len(rows),
        "candidate_observers": len(observers),
        "target_groups": int(sum(targets.values())),
        "independent_observer_counts": independent_counts,
        "insufficient_species": insufficient,
    }
    if insufficient or not rows:
        return [], {**base_audit, "feasible": False, "solver_status": "precheck_failed"}

    constraint_count = len(species) + len(observers)
    matrix = lil_matrix((constraint_count, len(rows)), dtype=np.float64)
    for column, row in enumerate(rows):
        matrix[species_index[int(row["taxon_id"])], column] = 1.0
        matrix[len(species) + observer_index[int(row["observer_id"])], column] = 1.0
    lower = np.concatenate(
        [np.asarray([targets[value] for value in species], dtype=np.float64), np.zeros(len(observers))]
    )
    upper = np.concatenate(
        [
            np.asarray([targets[value] for value in species], dtype=np.float64),
            np.full(len(observers), observer_capacity, dtype=np.float64),
        ]
    )
    costs = np.asarray(
        [
            stable_hash_int(seed, "milp-choice", row["taxon_id"], row["observation_id"]) / 2**64
            for row in rows
        ],
        dtype=np.float64,
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(rows), dtype=np.int8),
        bounds=Bounds(np.zeros(len(rows)), np.ones(len(rows))),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": float(time_limit_seconds)},
    )
    if result.x is None:
        return [], {
            **base_audit,
            "feasible": False,
            "solver_status": int(result.status),
            "solver_message": str(result.message),
        }
    rounded = np.rint(result.x)
    if not np.allclose(result.x, rounded, atol=1e-6):
        return [], {
            **base_audit,
            "feasible": False,
            "solver_status": int(result.status),
            "solver_message": "solver returned a fractional incumbent",
        }
    selected = [rows[index] for index, value in enumerate(rounded) if int(value) == 1]
    selected_counts = {
        str(taxon_id): sum(int(row["taxon_id"]) == taxon_id for row in selected) for taxon_id in species
    }
    selected_observers = [int(row["observer_id"]) for row in selected]
    feasible = (
        all(selected_counts[str(value)] == int(targets[value]) for value in species)
        and len(selected_observers) == len(set(selected_observers))
    )
    return selected if feasible else [], {
        **base_audit,
        "feasible": feasible,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "optimal": bool(result.success),
        "selected_groups": len(selected),
        "selected_observers": len(set(selected_observers)),
        "selected_counts": selected_counts,
    }


def select_bounded_total_groups(
    candidates: Iterable[Mapping[str, Any]],
    lower_targets: Mapping[int, int],
    upper_targets: Mapping[int, int],
    total_target: int,
    seed: int,
    *,
    observer_capacity: int = 1,
    time_limit_seconds: float = 300.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lower = {int(key): int(value) for key, value in lower_targets.items()}
    upper = {int(key): int(value) for key, value in upper_targets.items()}
    species = sorted(lower)
    if set(lower) != set(upper):
        raise ValueError("Lower and upper targets must cover identical taxa")
    if not species:
        raise ValueError("Bounded selection requires at least one taxon")
    if any(lower[value] < 0 or upper[value] < lower[value] for value in species):
        raise ValueError("Each taxon requires 0 <= lower target <= upper target")
    if observer_capacity < 1:
        raise ValueError("observer_capacity must be at least one")
    total_target = int(total_target)
    if not sum(lower.values()) <= total_target <= sum(upper.values()):
        raise ValueError("Exact total target must lie within the summed per-taxon bounds")

    rows = [dict(row) for row in candidates if int(row["taxon_id"]) in lower]
    observers = sorted({int(row["observer_id"]) for row in rows})
    species_index = {value: index for index, value in enumerate(species)}
    observer_index = {value: index for index, value in enumerate(observers)}
    independent_counts = {
        str(taxon_id): len(
            {int(row["observer_id"]) for row in rows if int(row["taxon_id"]) == taxon_id}
        )
        for taxon_id in species
    }
    insufficient = {
        str(taxon_id): {
            "available": independent_counts[str(taxon_id)],
            "lower_target": lower[taxon_id],
            "upper_target": upper[taxon_id],
        }
        for taxon_id in species
        if independent_counts[str(taxon_id)] < lower[taxon_id]
    }
    base_audit: dict[str, Any] = {
        "candidate_groups": len(rows),
        "candidate_observers": len(observers),
        "lower_targets": {str(key): lower[key] for key in species},
        "upper_targets": {str(key): upper[key] for key in species},
        "total_target": total_target,
        "independent_observer_counts": independent_counts,
        "insufficient_species": insufficient,
    }
    if insufficient or not rows:
        return [], {**base_audit, "feasible": False, "solver_status": "precheck_failed"}

    constraint_count = len(species) + len(observers) + 1
    matrix = lil_matrix((constraint_count, len(rows)), dtype=np.float64)
    total_row = constraint_count - 1
    for column, row in enumerate(rows):
        matrix[species_index[int(row["taxon_id"])], column] = 1.0
        matrix[len(species) + observer_index[int(row["observer_id"])], column] = 1.0
        matrix[total_row, column] = 1.0
    constraint_lower = np.concatenate(
        [
            np.asarray([lower[value] for value in species], dtype=np.float64),
            np.zeros(len(observers)),
            np.asarray([total_target], dtype=np.float64),
        ]
    )
    constraint_upper = np.concatenate(
        [
            np.asarray([upper[value] for value in species], dtype=np.float64),
            np.full(len(observers), observer_capacity, dtype=np.float64),
            np.asarray([total_target], dtype=np.float64),
        ]
    )
    costs = np.asarray(
        [
            stable_hash_int(seed, "bounded-milp-choice", row["taxon_id"], row["observation_id"])
            / 2**64
            for row in rows
        ],
        dtype=np.float64,
    )
    result = milp(
        c=costs,
        integrality=np.ones(len(rows), dtype=np.int8),
        bounds=Bounds(np.zeros(len(rows)), np.ones(len(rows))),
        constraints=LinearConstraint(matrix.tocsr(), constraint_lower, constraint_upper),
        options={"time_limit": float(time_limit_seconds)},
    )
    if result.x is None:
        return [], {
            **base_audit,
            "feasible": False,
            "solver_status": int(result.status),
            "solver_message": str(result.message),
        }
    rounded = np.rint(result.x)
    if not np.allclose(result.x, rounded, atol=1e-6):
        return [], {
            **base_audit,
            "feasible": False,
            "solver_status": int(result.status),
            "solver_message": "solver returned a fractional incumbent",
        }
    selected = [rows[index] for index, value in enumerate(rounded) if int(value) == 1]
    selected_counts = {
        str(taxon_id): sum(int(row["taxon_id"]) == taxon_id for row in selected)
        for taxon_id in species
    }
    observer_counts: dict[int, int] = {}
    for row in selected:
        observer_id = int(row["observer_id"])
        observer_counts[observer_id] = observer_counts.get(observer_id, 0) + 1
    feasible = (
        len(selected) == total_target
        and all(lower[value] <= selected_counts[str(value)] <= upper[value] for value in species)
        and all(count <= observer_capacity for count in observer_counts.values())
    )
    return selected if feasible else [], {
        **base_audit,
        "feasible": feasible,
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "optimal": bool(result.success),
        "selected_groups": len(selected),
        "selected_observers": len(observer_counts),
        "maximum_groups_per_selected_observer": max(observer_counts.values(), default=0),
        "selected_counts": selected_counts,
    }


def groups_to_canonical(
    groups: Iterable[Mapping[str, Any]], class_map: Mapping[int, int], cohort: str
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group in groups:
        taxon_id = int(group["taxon_id"])
        if taxon_id not in class_map:
            raise ValueError(f"Taxon {taxon_id} is absent from the frozen class map")
        place_ids = json.dumps(group.get("place_ids", []), separators=(",", ":"))
        for photo in group["photos"]:
            records.append(
                {
                    "cohort": cohort,
                    "observation_id": int(group["observation_id"]),
                    "observer_id": int(group["observer_id"]),
                    "taxon_id": taxon_id,
                    "class_index": int(class_map[taxon_id]),
                    "scientific_name": str(group["scientific_name"]),
                    "common_name": str(group.get("common_name", "")),
                    "observed_on": str(group.get("observed_on") or ""),
                    "created_at": str(group.get("created_at") or ""),
                    "place_ids": place_ids,
                    "photo_id": int(photo["photo_id"]),
                    "license_code": str(photo["license_code"]),
                    "attribution": str(photo.get("attribution", "")),
                    "url": str(photo["url"]),
                    "local_path": "",
                    "sha256": "",
                    "width": photo.get("width"),
                    "height": photo.get("height"),
                    "legacy_split": "",
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("Cannot convert an empty selected group list")
    frame = frame.sort_values(["class_index", "observation_id", "photo_id"]).reset_index(drop=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    return frame[CANONICAL_COLUMNS]


def build_frozen_taxa_frame(
    current_taxa: Iterable[Mapping[str, Any]], selected_new_taxa: Iterable[Mapping[str, Any]]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in sorted(current_taxa, key=lambda value: int(value["taxon_id"])):
        records.append(
            {
                "taxon_id": int(row["taxon_id"]),
                "scientific_name": str(row["scientific_name"]),
                "common_name": str(row.get("common_name", "")),
                "family": str(row.get("family", "")),
                "genus": str(row.get("genus", "")),
                "origin": "inspected_seed",
                "taxonomy_category_id": "",
                "existing_groups": 26,
                "new_group_target": 24,
            }
        )
    for row in selected_new_taxa:
        records.append(
            {
                "taxon_id": int(row["taxon_id"]),
                "scientific_name": str(row["scientific_name"]),
                "common_name": str(row.get("resolved_common_name") or row.get("common_name", "")),
                "family": str(row.get("family", "")),
                "genus": str(row.get("genus", "")),
                "origin": "inat2021_taxonomic_neighbor",
                "taxonomy_category_id": int(row["category_id"]),
                "existing_groups": 0,
                "new_group_target": 50,
            }
        )
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 100 or frame["taxon_id"].nunique() != 100:
        raise ValueError(f"Frozen taxonomy must contain 100 unique taxa, got {len(frame)} rows")
    frame.insert(0, "class_index", np.arange(len(frame), dtype=np.int64))
    return frame


def prepare_seed_manifest_for_taxonomy(
    seed_manifest_all: pd.DataFrame,
    class_map: Mapping[int, int],
    taxonomy_contract: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, set[int], set[int]]:
    excluded_observations = set(seed_manifest_all["observation_id"].astype(np.int64))
    excluded_observers = set(seed_manifest_all["observer_id"].astype(np.int64))
    seed_manifest = seed_manifest_all.copy()
    if taxonomy_contract:
        removed_id = int(taxonomy_contract["removed_seed_taxon_id"])
        removed = seed_manifest_all[seed_manifest_all["taxon_id"].astype(int) == removed_id]
        if removed["observation_id"].nunique() != 26:
            raise ValueError("Declared removed seed taxon does not contain exactly 26 seed groups")
        seed_manifest = seed_manifest_all[
            seed_manifest_all["taxon_id"].astype(int) != removed_id
        ].copy()
    seed_manifest["class_index"] = seed_manifest["taxon_id"].astype(int).map(class_map)
    if seed_manifest["class_index"].isna().any():
        raise ValueError("A retained seed taxon is absent from the frozen class map")
    seed_manifest["class_index"] = seed_manifest["class_index"].astype(np.int64)
    expected_seed_taxa = int((taxonomy_contract or {}).get("expected_seed_taxa", 80))
    if (
        seed_manifest["taxon_id"].nunique() != expected_seed_taxa
        or seed_manifest["observation_id"].nunique() != expected_seed_taxa * 26
    ):
        raise ValueError(
            f"Seed manifest does not match the frozen {expected_seed_taxa} species x 26 groups design"
        )
    return seed_manifest, excluded_observations, excluded_observers
