from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]


def test_v5_2_3_excludes_both_historical_removed_seed_taxa() -> None:
    config = (CODE_DIR / "configs" / "scaleup_replacement_v5_2_3.yaml").read_text(
        encoding="utf-8"
    )
    source = (CODE_DIR / "scripts" / "audit_scaleup_replacement_v5_2_3.py").read_text(
        encoding="utf-8"
    )
    assert "historical_removed_seed_taxon_id: 145224" in config
    assert "removed_ids =" in source
    assert "isin(removed_ids)" in source
    assert "replacement_audit_v5_2_3.json" in config
