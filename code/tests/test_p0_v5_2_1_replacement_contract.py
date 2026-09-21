from __future__ import annotations

from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parents[1]


def _load_config() -> dict:
    with (CODE_DIR / "configs" / "scaleup_replacement_v5_2_1.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        return yaml.safe_load(handle)


def test_v5_2_1_isolated_contract_preserves_scientific_values() -> None:
    config = _load_config()
    assert config["protocol"]["status"] == "frozen"
    assert config["seed"] == 20260819
    assert config["removed_seed_taxon_id"] == 145275
    assert config["selection"]["expected_new_groups"] == 2972
    assert config["selection"]["replacement_rule"].startswith("first_v5_1_pool_order")
    assert config["audit"]["expected_v5_1_reserve_feasible"] == 41
    assert config["audit"]["expected_v5_1_reserve_individually_feasible"] == 41


def test_v5_2_1_uses_isolated_outputs_and_binds_v5_2_failure() -> None:
    config = _load_config()
    required = set(config["protocol"]["freeze_required"])
    assert config["paths"]["attempts_json"].endswith("replacement_attempts_v5_2_1.json")
    assert config["paths"]["audit_json"].endswith("replacement_audit_v5_2_1.json")
    assert "../../FROZEN_DATASET_P0_V5_2_REPLACEMENT.sha256" in required
    assert "../scripts/audit_scaleup_replacement_v5_2_1.py" in required


def test_v5_2_1_has_no_download_or_model_branch() -> None:
    source = (CODE_DIR / "scripts" / "audit_scaleup_replacement_v5_2_1.py").read_text(
        encoding="utf-8"
    )
    assert "download_scaleup_images" not in source
    assert "train.py" not in source
    assert '"download_authorized": False' in source
