from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]


def test_v5_2_2_filters_unresolved_feasibility_rows() -> None:
    source = (CODE_DIR / "scripts" / "audit_scaleup_replacement_v5_2_2.py").read_text(
        encoding="utf-8"
    )
    assert 'if row.get("taxon_id") is not None' in source
    assert "replacement_attempts_v5_2_2.json" in (
        CODE_DIR / "configs" / "scaleup_replacement_v5_2_2.yaml"
    ).read_text(encoding="utf-8")
