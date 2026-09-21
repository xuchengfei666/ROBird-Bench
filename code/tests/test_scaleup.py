from __future__ import annotations

import pandas as pd

from robird.scaleup import (
    compact_candidates_by_observer,
    load_seed_identity,
    prepare_seed_manifest_for_taxonomy,
    rank_expansion_taxa,
    select_balanced_groups,
    select_bounded_total_groups,
)


def test_seed_identity_uses_original_scientific_name_when_confirmation_omits_it(tmp_path) -> None:
    original = pd.DataFrame.from_records(
        [
            {
                "observation_id": index,
                "observer_id": 100 + index,
                "taxon_id": 7,
                "species_name": "test bird",
                "scientific_name": "Testus birdus",
            }
            for index in range(1, 14)
        ]
    )
    confirmation = pd.DataFrame.from_records(
        [
            {
                "observation_id": 1000 + index,
                "observer_id": 2000 + index,
                "taxon_id": 7,
                "species_name": "test bird",
            }
            for index in range(1, 14)
        ]
    )
    original_path = tmp_path / "original.csv"
    confirmation_path = tmp_path / "confirmation.csv"
    original.to_csv(original_path, index=False)
    confirmation.to_csv(confirmation_path, index=False)

    taxa, observations, observers, audit = load_seed_identity(original_path, confirmation_path)
    assert taxa[0]["scientific_name"] == "Testus birdus"
    assert len(observations) == 26
    assert len(observers) == 26
    assert audit["groups_per_species"] == 26


def test_taxonomy_ranking_prefers_genus_then_family() -> None:
    current = [{"scientific_name": "Seedus alpha"}]
    categories = [
        {"id": 30, "name": "Otherus beta", "genus": "Otherus", "family": "Elseidae"},
        {"id": 20, "name": "Familyus beta", "genus": "Familyus", "family": "Seedidae"},
        {"id": 10, "name": "Seedus beta", "genus": "Seedus", "family": "Seedidae"},
        {"id": 1, "name": "Seedus alpha", "genus": "Seedus", "family": "Seedidae"},
    ]
    current[0]["family"] = "Seedidae"
    ranked = rank_expansion_taxa(categories, current)
    assert [row["scientific_name"] for row in ranked] == [
        "Seedus beta",
        "Familyus beta",
        "Otherus beta",
    ]


def test_candidate_compaction_keeps_one_group_per_observer() -> None:
    rows = [
        {"taxon_id": 1, "observer_id": 10, "observation_id": 100},
        {"taxon_id": 1, "observer_id": 10, "observation_id": 101},
        {"taxon_id": 1, "observer_id": 11, "observation_id": 102},
    ]
    compacted = compact_candidates_by_observer(rows, cap=10, seed=7, taxon_id=1)
    assert len(compacted) == 2
    assert len({row["observer_id"] for row in compacted}) == 2


def test_balanced_selection_enforces_global_observer_capacity() -> None:
    rows = [
        {"taxon_id": 1, "observer_id": 10, "observation_id": 100},
        {"taxon_id": 1, "observer_id": 11, "observation_id": 101},
        {"taxon_id": 1, "observer_id": 12, "observation_id": 102},
        {"taxon_id": 2, "observer_id": 10, "observation_id": 200},
        {"taxon_id": 2, "observer_id": 13, "observation_id": 201},
        {"taxon_id": 2, "observer_id": 14, "observation_id": 202},
    ]
    selected, audit = select_balanced_groups(rows, {1: 2, 2: 2}, seed=9)
    assert audit["feasible"]
    assert len(selected) == 4
    assert len({row["observer_id"] for row in selected}) == 4
    assert sum(row["taxon_id"] == 1 for row in selected) == 2
    assert sum(row["taxon_id"] == 2 for row in selected) == 2


def test_balanced_selection_reports_infeasible_species() -> None:
    selected, audit = select_balanced_groups(
        [{"taxon_id": 1, "observer_id": 10, "observation_id": 100}],
        {1: 2},
        seed=9,
    )
    assert selected == []
    assert not audit["feasible"]
    assert audit["insufficient_species"]["1"] == {"available": 1, "target": 2}


def test_bounded_selection_enforces_exact_total_bounds_and_observer_capacity() -> None:
    rows = [
        {"taxon_id": 1, "observer_id": 10, "observation_id": 100},
        {"taxon_id": 1, "observer_id": 11, "observation_id": 101},
        {"taxon_id": 1, "observer_id": 12, "observation_id": 102},
        {"taxon_id": 2, "observer_id": 10, "observation_id": 200},
        {"taxon_id": 2, "observer_id": 13, "observation_id": 201},
        {"taxon_id": 2, "observer_id": 14, "observation_id": 202},
    ]
    selected, audit = select_bounded_total_groups(
        rows,
        lower_targets={1: 1, 2: 1},
        upper_targets={1: 2, 2: 2},
        total_target=3,
        seed=9,
    )
    counts = {
        taxon_id: sum(int(row["taxon_id"]) == taxon_id for row in selected)
        for taxon_id in (1, 2)
    }
    assert audit["feasible"]
    assert len(selected) == 3
    assert all(1 <= counts[taxon_id] <= 2 for taxon_id in counts)
    assert len({row["observer_id"] for row in selected}) == 3
    assert audit["maximum_groups_per_selected_observer"] == 1


def test_bounded_selection_reports_infeasible_lower_bound() -> None:
    selected, audit = select_bounded_total_groups(
        [{"taxon_id": 1, "observer_id": 10, "observation_id": 100}],
        lower_targets={1: 2},
        upper_targets={1: 3},
        total_target=2,
        seed=9,
    )
    assert selected == []
    assert not audit["feasible"]
    assert audit["solver_status"] == "precheck_failed"
    assert audit["insufficient_species"]["1"] == {
        "available": 1,
        "lower_target": 2,
        "upper_target": 3,
    }


def test_repaired_taxonomy_drops_class_but_excludes_all_inspected_ids() -> None:
    rows = []
    for taxon_id, observer_offset in ((1, 100), (2, 200)):
        for index in range(26):
            rows.append(
                {
                    "taxon_id": taxon_id,
                    "observation_id": taxon_id * 1000 + index,
                    "observer_id": observer_offset + index,
                }
            )
    frame = pd.DataFrame(rows)
    retained, excluded_observations, excluded_observers = prepare_seed_manifest_for_taxonomy(
        frame,
        class_map={1: 0},
        taxonomy_contract={"removed_seed_taxon_id": 2, "expected_seed_taxa": 1},
    )
    assert set(retained["taxon_id"]) == {1}
    assert set(retained["class_index"]) == {0}
    assert len(excluded_observations) == 52
    assert len(excluded_observers) == 52
    assert {200, 225} <= excluded_observers
