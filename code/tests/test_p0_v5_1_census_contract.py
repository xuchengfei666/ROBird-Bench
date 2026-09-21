from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


CODE_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = CODE_DIR / "configs"
PROJECT_DIR = CODE_DIR.parent
SCRIPT_PATH = CODE_DIR / "scripts" / "audit_scaleup_census_v5_1.py"
LEGACY_SNAPSHOT = CODE_DIR / "results" / "dataset_p0" / "census_candidates_v5.json"
sys.path.insert(0, str(CODE_DIR / "src"))


def _load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_scaleup_census_v5_1", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_1_parser_accepts_only_known_empty_extension_forms() -> None:
    module = _load_script()
    assert (
        module.normalize_large_url_v5_1(
            "https://inaturalist-open-data.s3.amazonaws.com/photos/1273149/square.?x=1"
        )
        == "https://inaturalist-open-data.s3.amazonaws.com/photos/1273149/large.?x=1"
    )
    assert module.normalize_large_url_v5_1("https://example.test/small.jpg") == "https://example.test/large.jpg"
    for malformed in (
        "https://example.test/photo.jpg",
        "https://example.test/unknown.",
        "https://example.test/square",
    ):
        try:
            module.normalize_large_url_v5_1(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Malformed URL was accepted: {malformed}")


def test_v5_1_source_contract_only_adds_parser_rule() -> None:
    v5 = _load_yaml("scaleup_census_v5.yaml")
    v5_1 = _load_yaml("scaleup_census_v5_1.yaml")
    source = dict(v5_1["source"])
    assert source.pop("url_normalization") == "allow_known_size_empty_extension_to_large"
    assert source == v5["source"]
    assert v5_1["paths"]["legacy_v5_snapshot_json"].endswith("census_candidates_v5.json")
    assert v5_1["paths"]["census_snapshot_json"].endswith("census_candidates_v5_1.json")
    assert v5_1["paths"]["census_audit_json"].endswith("census_audit_v5_1.json")


def test_v5_1_binds_the_immutable_86_taxon_checkpoint() -> None:
    config = _load_yaml("scaleup_census_v5_1.yaml")
    checkpoint = json.loads(LEGACY_SNAPSHOT.read_text(encoding="utf-8"))
    bootstrap = config["bootstrap"]
    assert checkpoint["status"] == bootstrap["expected_legacy_v5_status"] == "RUNNING"
    assert checkpoint["contract_hash"] == bootstrap["expected_legacy_v5_contract_hash"]
    assert len(checkpoint["completed_taxa"]) == bootstrap["expected_completed_taxa"] == 86
    assert len(checkpoint["groups"]) == bootstrap["expected_candidate_groups"] == 7532
    assert checkpoint["completed_taxa"][-1] == bootstrap["expected_last_taxon_id"] == 6988


def test_v5_1_has_no_dataset_decision_or_download_path() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "select_bounded_total_groups" not in source
    assert "download_scaleup_images" not in source
    assert "PASS_METADATA_TO_DOWNLOAD" not in source
    assert "audit_scaleup_census_v5.py" in source
    protocol = (PROJECT_DIR / "DATASET_P0_V5_1_CENSUS_PROTOCOL.md").read_text(encoding="utf-8")
    assert "must not create a v5.1 class table" in protocol
