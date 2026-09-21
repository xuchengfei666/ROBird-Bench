from __future__ import annotations

from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_p0_v2_changes_only_metadata_page_depth() -> None:
    v1 = _load("scaleup_p0.yaml")
    v2 = _load("scaleup_p0_v2.yaml")

    source_v1 = dict(v1["source"])
    source_v2 = dict(v2["source"])
    assert source_v1.pop("pages") == 5
    assert source_v2.pop("pages") == 20
    assert source_v1 == source_v2
    assert v1["seed"] == v2["seed"]
    assert v1["selection"] == v2["selection"]
    assert v1["download"] == v2["download"]


def test_p0_v2_dataset_gates_match_v1_and_outputs_are_isolated() -> None:
    dataset_v1 = _load("dataset_p0.yaml")
    dataset_v2 = _load("dataset_p0_v2.yaml")
    scaleup_v1 = _load("scaleup_p0.yaml")
    scaleup_v2 = _load("scaleup_p0_v2.yaml")

    for key in ("seed", "licenses", "eligibility", "split", "gates", "holdout"):
        assert dataset_v1[key] == dataset_v2[key]
    for key in (
        "candidate_snapshot_json",
        "selection_audit_json",
        "metadata_manifest_csv",
        "canonical_manifest_csv",
        "download_root",
        "download_ledger_json",
    ):
        assert scaleup_v1["paths"][key] != scaleup_v2["paths"][key]
    for key in (
        "canonical_manifest_csv",
        "audit_json",
        "split_csv",
        "split_audit_json",
    ):
        assert dataset_v1["paths"][key] != dataset_v2["paths"][key]
    assert dataset_v1["duplicates"]["candidate_csv"] != dataset_v2["duplicates"]["candidate_csv"]
    assert dataset_v1["duplicates"]["candidate_meta_json"] != dataset_v2["duplicates"]["candidate_meta_json"]
