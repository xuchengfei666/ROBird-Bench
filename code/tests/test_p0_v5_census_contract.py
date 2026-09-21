from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml


CODE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = CODE_DIR / "configs"
PROJECT_DIR = CODE_DIR.parent
SCRIPT_PATH = CODE_DIR / "scripts" / "audit_scaleup_census_v5.py"


def _load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_scaleup_census_v5", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_census_source_is_exactly_v4_and_has_no_execution_branch() -> None:
    census = _load_yaml("scaleup_census_v5.yaml")
    v4 = _load_yaml("scaleup_p0_v4.yaml")
    assert census["source"] == v4["source"]
    assert census["seed"] == v4["seed"]
    assert "download" not in census
    assert "selection" not in census
    assert set(census["paths"]) == {
        "original_manifest_csv",
        "original_source_json",
        "original_download_ledger",
        "confirmation_manifest_csv",
        "confirmation_source_json",
        "confirmation_download_ledger",
        "feasibility_audit_json",
        "frozen_taxa_v4_csv",
        "v4_candidate_snapshot_json",
        "v4_selection_audit_json",
        "census_snapshot_json",
        "census_audit_json",
    }


def test_census_pool_is_exact_current_plus_all_frozen_feasible_reserves() -> None:
    module = _load_script()
    config = _load_yaml("scaleup_census_v5.yaml")
    taxa = pd.read_csv(CODE_DIR / "data" / "manifests" / "frozen_100_taxa_v4.csv")
    feasibility = json.loads(
        (CODE_DIR / "results" / "scaleup_feasibility" / "audit.json").read_text(
            encoding="utf-8"
        )
    )
    pool = module.build_taxon_pool(taxa, feasibility, config["census"])
    current = [row for row in pool if row["pool"] == "current_v4"]
    reserve = [row for row in pool if row["pool"] == "reserve"]
    assert len(pool) == len({row["taxon_id"] for row in pool}) == 141
    assert len(current) == 100
    assert len(reserve) == 41
    assert sum(row["lower_target"] == 22 for row in current) == 79
    assert sum(row["lower_target"] == 48 for row in current) == 21
    assert {row["lower_target"] for row in reserve} == {48}
    assert reserve[0]["taxon_id"] == 4478
    assert reserve[0]["scientific_name"] == "Sterna striata"


def test_census_bootstrap_is_exact_terminal_v4_prefix() -> None:
    module = _load_script()
    config = _load_yaml("scaleup_census_v5.yaml")
    taxa = pd.read_csv(CODE_DIR / "data" / "manifests" / "frozen_100_taxa_v4.csv")
    feasibility = json.loads(
        (CODE_DIR / "results" / "scaleup_feasibility" / "audit.json").read_text(
            encoding="utf-8"
        )
    )
    pool = module.build_taxon_pool(taxa, feasibility, config["census"])
    v4_snapshot = json.loads(
        (CODE_DIR / "results" / "dataset_p0" / "scaleup_candidates_v4.json").read_text(
            encoding="utf-8"
        )
    )
    v4_audit = json.loads(
        (CODE_DIR / "results" / "dataset_p0" / "scaleup_selection_audit_v4.json").read_text(
            encoding="utf-8"
        )
    )
    groups, completed, records = module.bootstrap_v4_prefix(
        pool, v4_snapshot, v4_audit, config
    )
    assert len(completed) == len(records) == 71
    assert len(groups) == 6050
    assert completed == [row["taxon_id"] for row in pool[:71]]
    assert records[-1]["scientific_name"] == "Cardellina canadensis"
    assert records[-1]["shortfall"] == 2
    assert {row["record_source"] for row in records} == {"v4_hash_verified_bootstrap"}


def test_census_cannot_emit_dataset_pass_or_run_selection() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "PASS_METADATA_TO_DOWNLOAD" not in source
    assert "select_bounded_total_groups" not in source
    assert "download_scaleup_images" not in source
    assert '"dataset_p0_decision_authorized": False' in source
    assert '"download_authorized": False' in source


def test_census_freeze_contract_binds_v4_terminal_evidence_and_own_script() -> None:
    config = _load_yaml("scaleup_census_v5.yaml")
    required = set(config["protocol"]["freeze_required"])
    assert "../scripts/audit_scaleup_census_v5.py" in required
    assert "../../FROZEN_DATASET_P0_V4.sha256" in required
    assert "../results/dataset_p0/scaleup_candidates_v4.json" in required
    assert "../results/dataset_p0/scaleup_selection_audit_v4.json" in required
    protocol = (PROJECT_DIR / "DATASET_P0_V5_CENSUS_PROTOCOL.md").read_text(encoding="utf-8")
    assert "cannot produce a download authorization" in protocol
    assert "141 unique taxa" in protocol
