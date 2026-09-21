from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


CODE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = CODE_DIR / "configs"
PROJECT_DIR = CODE_DIR.parent


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_p0_v4_changes_only_taxonomy_and_new_group_accounting_from_v3() -> None:
    v3 = _load("scaleup_p0_v3.yaml")
    v4 = _load("scaleup_p0_v4.yaml")
    assert v4["source"] == v3["source"]
    assert v4["seed"] == v3["seed"]
    assert v4["download"] == v3["download"]
    for key in (
        "mode",
        "existing_groups_per_species",
        "final_groups_per_species_min",
        "final_groups_per_species_max",
        "candidate_multiplier",
        "max_groups_per_new_observer",
        "expected_species",
        "expected_groups",
    ):
        assert v4["selection"][key] == v3["selection"][key]
    assert v3["selection"]["new_groups_total"] == 2920
    assert v4["selection"]["new_groups_total"] == 2946
    assert v4["taxonomy_contract"] == {
        "removed_seed_taxon_id": 145224,
        "removed_seed_scientific_name": "Geothlypis philadelphia",
        "replacement_taxon_id": 4449,
        "replacement_scientific_name": "Sterna paradisaea",
        "replacement_source_rule": "first_unused_feasible_candidate_in_frozen_audit_order",
        "expected_seed_taxa": 79,
        "expected_expansion_taxa": 21,
    }


def test_p0_v4_table_is_exact_deterministic_replacement() -> None:
    v3 = pd.read_csv(CODE_DIR / "data" / "manifests" / "frozen_100_taxa.csv")
    v4 = pd.read_csv(CODE_DIR / "data" / "manifests" / "frozen_100_taxa_v4.csv")
    assert len(v4) == v4["taxon_id"].nunique() == 100
    assert list(v4["class_index"]) == list(range(100))
    assert set(v3["taxon_id"].astype(int)) - set(v4["taxon_id"].astype(int)) == {145224}
    assert set(v4["taxon_id"].astype(int)) - set(v3["taxon_id"].astype(int)) == {4449}
    assert (v4["origin"] == "inspected_seed").sum() == 79
    assert (v4["origin"] == "inat2021_taxonomic_neighbor").sum() == 21
    assert int(v4["existing_groups"].sum()) == 79 * 26
    replacement = v4[v4["taxon_id"].astype(int) == 4449].iloc[0]
    assert replacement["scientific_name"] == "Sterna paradisaea"
    assert replacement["origin"] == "inat2021_taxonomic_neighbor"
    assert int(replacement["existing_groups"]) == 0


def test_p0_v4_replacement_is_first_unused_feasible_audit_candidate() -> None:
    feasibility = json.loads(
        (CODE_DIR / "results" / "scaleup_feasibility" / "audit.json").read_text(encoding="utf-8")
    )
    v3 = pd.read_csv(CODE_DIR / "data" / "manifests" / "frozen_100_taxa.csv")
    v3_ids = set(v3["taxon_id"].astype(int))
    fallback = next(
        row
        for row in feasibility["candidates"]
        if row.get("resolved")
        and int(row["taxon_id"]) not in v3_ids
        and int(row["eligible_groups"]) >= 50
        and int(row["eligible_observers"]) >= 50
    )
    assert int(fallback["taxon_id"]) == 4449
    assert fallback["scientific_name"] == "Sterna paradisaea"
    assert int(fallback["eligible_groups"]) == 153
    assert int(fallback["eligible_observers"]) == 87


def test_p0_v4_dataset_gates_match_v3_and_paths_are_isolated() -> None:
    dataset_v3 = _load("dataset_p0_v3.yaml")
    dataset_v4 = _load("dataset_p0_v4.yaml")
    scaleup_v3 = _load("scaleup_p0_v3.yaml")
    scaleup_v4 = _load("scaleup_p0_v4.yaml")
    for key in ("seed", "licenses", "eligibility", "split", "gates", "holdout"):
        assert dataset_v4[key] == dataset_v3[key]
    for key in (
        "candidate_snapshot_json",
        "selection_audit_json",
        "metadata_manifest_csv",
        "canonical_manifest_csv",
        "download_root",
        "download_ledger_json",
    ):
        assert scaleup_v4["paths"][key] != scaleup_v3["paths"][key]
    assert dataset_v4["paths"] != dataset_v3["paths"]
    assert dataset_v4["duplicates"] != dataset_v3["duplicates"]


def test_p0_v4_freeze_contract_binds_all_prior_stop_audits() -> None:
    for name in ("dataset_p0_v4.yaml", "scaleup_p0_v4.yaml"):
        required = set(_load(name)["protocol"]["freeze_required"])
        assert "../results/dataset_p0/scaleup_selection_audit.json" in required
        assert "../results/dataset_p0/scaleup_selection_audit_v2.json" in required
        assert "../results/dataset_p0/scaleup_selection_audit_v3.json" in required
        assert "../results/dataset_p0/scaleup_candidates_v3.json" in required
