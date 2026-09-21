from __future__ import annotations

import pandas as pd
import pytest

from robird.manifest import combine_seed_manifests, validate_manifest_schema


def _legacy_rows(cohort: str, taxon_id: int, observation_id: int, photo_ids: list[int]) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "cohort": cohort,
                "observation_id": observation_id,
                "observer_id": observation_id + 100,
                "taxon_id": taxon_id,
                "scientific_name": f"Species {taxon_id}",
                "common_name": f"Bird {taxon_id}",
                "observed_on": "2026-01-01",
                "created_at": "2026-01-01T00:00:00Z",
                "place_ids": "[]",
                "photo_id": photo_id,
                "license_code": "cc-by",
                "attribution": "Test observer",
                "url": f"https://example.invalid/{photo_id}",
                "local_path": f"E:/test/{photo_id}.jpg",
                "sha256": f"hash-{photo_id}",
                "width": 100,
                "height": 100,
                "legacy_split": "legacy",
            }
            for photo_id in photo_ids
        ]
    )


def test_combination_assigns_stable_contiguous_classes_and_rows() -> None:
    original = _legacy_rows("original", 20, 2, [21, 22])
    confirmation = _legacy_rows("confirmation", 10, 1, [11, 12])
    combined = combine_seed_manifests(original, confirmation)

    assert combined["row_id"].tolist() == list(range(4))
    assert combined.loc[combined["taxon_id"] == 10, "class_index"].unique().tolist() == [0]
    assert combined.loc[combined["taxon_id"] == 20, "class_index"].unique().tolist() == [1]
    assert validate_manifest_schema(combined) == []


def test_combination_rejects_photo_overlap_between_cohorts() -> None:
    original = _legacy_rows("original", 10, 1, [11, 12])
    confirmation = _legacy_rows("confirmation", 20, 2, [12, 13])
    with pytest.raises(ValueError, match="overlap"):
        combine_seed_manifests(original, confirmation)


def test_schema_reports_group_inconsistency() -> None:
    combined = combine_seed_manifests(
        _legacy_rows("original", 10, 1, [11, 12]),
        _legacy_rows("confirmation", 20, 2, [21, 22]),
    )
    combined.loc[combined["photo_id"] == 12, "observer_id"] = 999
    violations = validate_manifest_schema(combined)
    assert "observation 1 has multiple observers" in violations
