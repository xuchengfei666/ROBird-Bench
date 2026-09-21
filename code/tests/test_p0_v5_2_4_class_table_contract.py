from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parents[1]


def test_class_table_freeze_is_bound_to_v5_2_4_pass() -> None:
    with (CODE_DIR / "configs" / "scaleup_class_table_v5_2_4.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = set(config["protocol"]["freeze_required"])
    assert "../../FROZEN_DATASET_P0_V5_2_4_REPLACEMENT.sha256" in required
    assert "../results/dataset_p0/replacement_audit_v5_2_4.json" in required
    assert config["selection"]["replacement_taxon_id"] == 4478
    assert config["selection"]["expected_seed_taxa"] == 78
    assert config["selection"]["expected_expansion_taxa"] == 22
