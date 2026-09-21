"""Synthetic-only rater fixtures; never create labels for real cohort images."""
import importlib.util
import json
import threading
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import pytest
from robird.io import atomic_write_json
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('quality_review',ROOT/'code/scripts/quality_review_v1.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


@pytest.fixture
def store(tmp_path):
    queue=dict(items=[dict(item_id='synthetic_0',photo_id=-1,sha256='test-hash',local_path='does-not-exist',width=20,height=20,bbox=None)],orders={'A':[0],'B':[0]})
    atomic_write_json(tmp_path/'queue_private.json',queue)
    return q.ReviewStore(tmp_path)


def answer():
    return dict(item_id='synthetic_0',sha256='test-hash',rater_id='SYNTHETIC_TEST_RATER',reviewer_type='human',independent_attestation=True,
        bbox_valid='not_detected',motion_blur_present='uncertain',occlusion_fraction='uncertain',background_complexity_1_5='uncertain',
        confidence_1_5=1,target_ambiguous='uncertain',notes='SYNTHETIC UNIT TEST ONLY')


def test_empty_package_not_complete_and_no_autofill(store):
    assert store.export('A')['completed_items']==0
    assert store.item('A',0)['saved'] is None
    with pytest.raises(q.GateStop):q.validate_export(store.export('A'),store,'A')


@pytest.mark.parametrize('field,value',[('reviewer_type','ai'),('independent_attestation',False),('occlusion_fraction',1.2),('bbox_valid','yes'),('confidence_1_5',0),('sha256','wrong')])
def test_rejects_invalid_or_nonhuman_labels(store,field,value):
    row=answer();row[field]=value
    with pytest.raises(ValueError):store.save('A',row)


def test_versioned_synthetic_saves_and_full_validation(store):
    store.save('A',answer());store.save('A',dict(answer(),notes='Updated synthetic fixture'))
    assert len(list((store.root/'annotations/A/versions/synthetic_0').glob('*.json')))==2
    validated=q.validate_export(store.export('A'),store,'A')
    assert len(validated)==1 and validated.occlusion_fraction.iloc[0]=='uncertain'
    assert store.export('B')['completed_items']==0


def test_http_same_origin_and_path_boundary(store):
    server=q.ThreadingHTTPServer(('127.0.0.1',0),q.make_handler(store));thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base+'/api/item?slot=A&position=0') as r:assert json.load(r)['total']==1
        with pytest.raises(HTTPError):urlopen(base+'/image/../../secrets')
        req=Request(base+'/api/save?slot=A',data=json.dumps(answer()).encode(),headers={'Content-Type':'application/json'})
        with pytest.raises(HTTPError):urlopen(req)
        req.add_header('Origin',base)
        with urlopen(req) as r:assert json.load(r)['status']=='SAVED'
        req=Request(base+'/',headers={'Host':'evil.example'})
        with pytest.raises(HTTPError):urlopen(req)
    finally:server.shutdown();server.server_close();thread.join()
