from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_p0_v3_keeps_v2_source_and_changes_only_balance_contract() -> None:
    v2 = _load("scaleup_p0_v2.yaml")
    v3 = _load("scaleup_p0_v3.yaml")

    assert v3["source"] == v2["source"]
    assert v3["seed"] == v2["seed"]
    assert v3["download"] == v2["download"]
    for key in (
        "candidate_multiplier",
        "max_groups_per_new_observer",
        "expected_species",
        "expected_groups",
    ):
        assert v3["selection"][key] == v2["selection"][key]
    assert v3["selection"] == {
        "mode": "bounded_total",
        "existing_groups_per_species": 26,
        "final_groups_per_species_min": 48,
        "final_groups_per_species_max": 52,
        "new_groups_total": 2920,
        "candidate_multiplier": 4,
        "max_groups_per_new_observer": 1,
        "expected_species": 100,
        "expected_groups": 5000,
    }


def test_p0_v3_dataset_gates_match_v2_and_outputs_are_isolated() -> None:
    dataset_v2 = _load("dataset_p0_v2.yaml")
    dataset_v3 = _load("dataset_p0_v3.yaml")
    scaleup_v2 = _load("scaleup_p0_v2.yaml")
    scaleup_v3 = _load("scaleup_p0_v3.yaml")

    for key in ("seed", "licenses", "eligibility", "split", "gates", "holdout"):
        assert dataset_v3[key] == dataset_v2[key]
    input_paths = (
        "original_manifest_csv",
        "original_source_json",
        "original_download_ledger",
        "confirmation_manifest_csv",
        "confirmation_source_json",
        "confirmation_download_ledger",
        "feasibility_audit_json",
        "frozen_taxa_csv",
    )
    for key in input_paths:
        assert scaleup_v3["paths"][key] == scaleup_v2["paths"][key]
    for key in (
        "candidate_snapshot_json",
        "selection_audit_json",
        "metadata_manifest_csv",
        "canonical_manifest_csv",
        "download_root",
        "download_ledger_json",
    ):
        assert scaleup_v3["paths"][key] != scaleup_v2["paths"][key]
    assert dataset_v3["paths"] != dataset_v2["paths"]
    assert dataset_v3["duplicates"] != dataset_v2["duplicates"]


def test_p0_v3_freeze_binds_both_prior_stop_audits() -> None:
    for name in ("dataset_p0_v3.yaml", "scaleup_p0_v3.yaml"):
        config = _load(name)
        required = set(config["protocol"]["freeze_required"])
        assert "../results/dataset_p0/scaleup_selection_audit.json" in required
        assert "../results/dataset_p0/scaleup_selection_audit_v2.json" in required
