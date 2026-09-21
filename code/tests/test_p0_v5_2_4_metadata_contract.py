from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parents[1]


def test_v5_2_4_metadata_config_binds_class_table_and_downloader() -> None:
    with (CODE_DIR / "configs" / "scaleup_p0_v5_2_4.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = set(config["protocol"]["freeze_required"])
    assert "../../FROZEN_DATASET_P0_V5_2_4_CLASS_TABLE.sha256" in required
    assert "../scripts/download_scaleup_images.py" in required
    assert config["selection"]["expected_groups"] == 5000
    assert config["selection"]["expected_new_groups"] == 2972
