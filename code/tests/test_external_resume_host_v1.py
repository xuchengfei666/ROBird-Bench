import importlib.util
import hashlib
import json
from pathlib import Path
import pytest

SPEC=importlib.util.spec_from_file_location('resume_host',Path(__file__).resolve().parents[1]/'scripts/run_external_resume_host_v1.py')
host=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(host)


def test_exit_status_never_confuses_stale_running_with_completion():
    assert host.classify_exit(0,{'status':'RUNNING'})=='UNEXPECTED_CHILD_EXIT'
    assert host.classify_exit(1,{'status':'RUNNING'})=='UNEXPECTED_CHILD_EXIT'
    assert host.classify_exit(0,{'status':'STOP_DATA_OR_REVIEW_GATE'})=='CHILD_RECORDED_STOP'
    assert host.classify_exit(0,{'status':'AUTOMATIC_COMPLETE_HUMAN_QUALITY_ANNOTATION_PENDING'})=='AUTOMATIC_DONE_HUMAN_ANNOTATIONS_PENDING'


def test_ledger_integrity_no_download_or_mutation(tmp_path):
    image=tmp_path/'photo.bin';image.write_bytes(b'original')
    digest=hashlib.sha256(b'original').hexdigest()
    ledger=tmp_path/'ledgers';ledger.mkdir()
    host.write(ledger/'123.json',dict(contract='contract',artifacts={str(image):digest},details=dict(sha256=digest)))
    before=(ledger/'123.json').read_bytes()
    assert len(host.validate_ledgers(ledger,'contract'))==1
    assert (ledger/'123.json').read_bytes()==before
    with pytest.raises(RuntimeError):host.validate_ledgers(ledger,'other')
    image.write_bytes(b'changed')
    with pytest.raises(RuntimeError):host.validate_ledgers(ledger,'contract')
