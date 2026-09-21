from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


CODE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = CODE_DIR / "configs"
PROJECT_DIR = CODE_DIR.parent
SCRIPT_PATH = CODE_DIR / "scripts" / "audit_scaleup_replacement_v5_2.py"
sys.path.insert(0, str(CODE_DIR / "src"))


def _load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_scaleup_replacement_v5_2", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_2_accounting_and_source_boundary_are_frozen() -> None:
    config = _load_yaml("scaleup_replacement_v5_2.yaml")
    v5_1 = _load_yaml("scaleup_census_v5_1.yaml")
    assert config["seed"] == v5_1["seed"] == 20260819
    assert config["selection"]["expected_current_seed_taxa"] == 78
    assert config["selection"]["expected_expansion_taxa"] == 22
    assert config["selection"]["expected_new_groups"] == 2972
    assert config["selection"]["final_groups_per_species_min"] == 48
    assert config["selection"]["final_groups_per_species_max"] == 52
    assert config["selection"]["replacement_rule"].startswith("first_v5_1_pool_order")
    assert "download" not in config


def test_v5_2_first_reserve_is_scheduled_from_complete_census() -> None:
    audit = json.loads(
        (CODE_DIR / "results" / "dataset_p0" / "census_audit_v5_1.json").read_text(
            encoding="utf-8"
        )
    )
    reserves = sorted(audit["reserve_individually_feasible"], key=lambda row: int(row["census_index"]))
    assert len(reserves) == 41
    assert reserves[0]["taxon_id"] == 4478
    assert reserves[0]["scientific_name"] == "Sterna striata"
    assert all(int(row["shortfall"]) == 0 for row in reserves)


def test_v5_2_temporary_table_has_78_seed_and_22_expansion_taxa() -> None:
    module = _load_script()
    config = _load_yaml("scaleup_replacement_v5_2.yaml")
    table = pd.read_csv(CODE_DIR / "data" / "manifests" / "frozen_100_taxa_v4.csv")
    feasibility = json.loads(
        (CODE_DIR / "results" / "scaleup_feasibility" / "audit.json").read_text(
            encoding="utf-8"
        )
    )
    replacement = json.loads(
        (CODE_DIR / "results" / "dataset_p0" / "census_audit_v5_1.json").read_text(
            encoding="utf-8"
        )
    )["reserve_individually_feasible"][0]
    feasibility_row = next(
        row for row in feasibility["candidates"] if int(row["taxon_id"]) == int(replacement["taxon_id"])
    )
    temporary = module._replace_table(
        table, replacement, feasibility_row, int(config["removed_seed_taxon_id"])
    )
    lower, upper, total = module._targets(temporary, config["selection"])
    assert len(temporary) == temporary["taxon_id"].nunique() == 100
    assert int((temporary["origin"] == "inspected_seed").sum()) == 78
    assert int((temporary["origin"] == "inat2021_taxonomic_neighbor").sum()) == 22
    assert sum(temporary["existing_groups"].astype(int)) == 78 * 26
    assert total == 2972
    assert set(lower.values()) == {22, 48}
    assert set(upper.values()) == {26, 52}


def test_v5_2_script_has_no_download_or_model_authority() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "download_scaleup_images" not in source
    assert "train.py" not in source
    assert "build_scaleup_manifest" not in source
    assert '"download_authorized": False' in source
    assert '"class_table_written": False' in source
    assert '"metadata_manifest_written": False' in source


def test_v5_2_freeze_binds_complete_v5_1_evidence() -> None:
    config = _load_yaml("scaleup_replacement_v5_2.yaml")
    required = set(config["protocol"]["freeze_required"])
    assert "../results/dataset_p0/census_candidates_v5_1.json" in required
    assert "../results/dataset_p0/census_audit_v5_1.json" in required
    assert "../../FROZEN_DATASET_P0_V5_1_CENSUS.sha256" in required
    protocol = (PROJECT_DIR / "DATASET_P0_V5_2_REPLACEMENT_PROTOCOL.md").read_text(encoding="utf-8")
    assert "first candidate passing all gates" in protocol
    assert "2,972" in protocol
