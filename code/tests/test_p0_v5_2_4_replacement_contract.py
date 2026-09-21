from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]


def test_v5_2_4_checks_final_and_new_group_bounds() -> None:
    source = (CODE_DIR / "scripts" / "audit_scaleup_replacement_v5_2_4.py").read_text(
        encoding="utf-8"
    )
    assert "seed_group_counts" in source
    assert "48 <= group_counts.get(taxon_id, 0) <= 52" in source
    assert "group_counts.get(taxon_id, 0) - seed_group_counts.get(taxon_id, 0)" in source
