from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parents[1]


def test_v5_2_4_byte_audit_binds_manifest_and_e_ledger() -> None:
    with (CODE_DIR / "configs" / "dataset_p0_v5_2_4.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = set(config["protocol"]["freeze_required"])
    assert "../data/manifests/development_photos_v5_2_4.csv" in required
    assert "E:/Datasets/ROBird-Bench/p0-v5_2_4/metadata/download_ledger.json" in required
    assert config["eligibility"]["min_groups_per_species"] == 48
    assert config["duplicates"]["hamming_threshold"] == 4
