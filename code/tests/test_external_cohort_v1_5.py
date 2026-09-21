import copy
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from robird import external_cohort_v1_5 as m
from robird.io import atomic_write_csv

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def review_inputs():
    # Exact real pair identities without repeated disk image I/O in unit tests.
    old=pd.read_csv(ROOT/'code/results/external_cohort_v1_3/duplicate_review.csv',dtype=str,keep_default_na=False)
    current=pd.read_csv(ROOT/'code/results/external_cohort_v1_4/duplicate_review.csv',dtype=str,keep_default_na=False)
    index=m.read_json(m.INDEX)
    notes=m.read_json(ROOT/'code/configs/external_duplicate_decisions_v1_5.json')
    hashes={}
    for row,item in zip(current.to_dict('records'),index['pairs']):
        for key in ('path_a','path_b'):
            hashes[str(Path(row[key]))]=item[key+'_sha256']
            hashes[str(Path(item[key]))]=item[key+'_sha256']
    return current,old,index,notes,lambda p:hashes[str(p)]


def test_review_all_65_exact_real_identities(review_inputs):
    result=m.bind_review(*review_inputs)
    assert len(result)==65
    assert dict(m.Counter(r['decision'] for r in result))==m.COUNTS
    assert all(len(r['path_a_sha256'])==64 for r in result)


@pytest.mark.parametrize('change',['missing','duplicate','row','hash','exact','rationale','human'])
def test_review_fails_closed(review_inputs,change):
    current,old,index,notes,hasher=review_inputs
    if change=='missing':notes['pairs'].pop()
    if change=='duplicate':notes['pairs'][-1]=copy.deepcopy(notes['pairs'][0])
    if change=='row':current.loc[0,'photo_id']='123'
    if change=='hash':hasher=lambda p:'f'*64
    if change=='exact':current.loc[1,'exact_bytes']='True';old.loc[1,'exact_bytes']='True';index['pairs'][1]['exact_bytes']='True'
    if change=='rationale':notes['pairs'][1]['rationale']=''
    if change=='human':notes['independent_human_adjudication']=True
    with pytest.raises(m.GateStop):m.bind_review(current,old,index,notes,hasher)


@pytest.fixture
def shared_frame():
    data=[]
    for obs,taxon,photos in [(377824923,4328,[690953732,690960036,690960029]),
                              (377824924,4381,[690953732,690959646,690959649])]:
        for i,p in enumerate(photos):
            data.append(dict(observation_id=obs,taxon_id=taxon,photo_id=p,observer_id=2132783,sha256=str(i)*64))
    data += [dict(observation_id=9,taxon_id=4328,photo_id=p,observer_id=999,sha256='x'+str(p)) for p in (1,2)]
    return pd.DataFrame(data)


def test_cluster_exclusion_keeps_source_and_feature_alignment(shared_frame):
    original=shared_frame.copy(deep=True)
    features=np.arange(8*384).reshape(8,384)
    result,aligned=m.exclude_cluster(shared_frame,features)
    assert result.observation_id.tolist()==[9,9]
    np.testing.assert_array_equal(aligned,features[6:])
    pd.testing.assert_frame_equal(shared_frame,original)
    with pytest.raises(m.GateStop):m.exclude_cluster(shared_frame,features[:7])
    changed=shared_frame.copy();changed.loc[4,'sha256']='changed'
    with pytest.raises(m.GateStop):m.exclude_cluster(changed)


def test_evaluate_both_calls_real_production_mask(monkeypatch,shared_frame,tmp_path):
    calls=[]
    monkeypatch.setattr(m.legacy,'evaluate_external',lambda f,x,c,r,d,p,k:calls.append((f.copy(),x.copy(),d,p)))
    features=np.arange(8*384).reshape(8,384)
    m.evaluate_both(shared_frame,features,dict(runs=[]),tmp_path,tmp_path/'data',tmp_path/'report','test')
    assert [len(c[0]) for c in calls]==[8,2]
    np.testing.assert_array_equal(calls[1][1],features[6:])
    assert calls[1][3].name=='sensitivity_no_cluster'


def test_shared_diagnostic_semicolon_and_all_subsets(shared_frame,tmp_path):
    rows=[]
    for obs,part in shared_frame[shared_frame.observation_id.isin(m.SHARED_OBS)].groupby('observation_id'):
        for k in (1,2,3):
            for subset in itertools.combinations(part.photo_id,k):
                p=np.zeros(100);p[0]=1
                rows.append(dict(observation_id=obs,budget=k,photo_ids=';'.join(map(str,subset)),
                    label=0 if obs==377824923 else 1,prediction=0,probabilities_json=json.dumps(p.tolist())))
    data=tmp_path/'data';report=tmp_path/'report'
    atomic_write_csv(data/'runs/test/predictions.csv',pd.DataFrame(rows))
    m.shared_diagnostic(shared_frame,data,report,dict(runs=[dict(run_id='test')]),'test')
    result=pd.read_csv(report/'shared_inputs_diagnostic.csv')
    assert len(result)==14 and result.correct.sum()==7
    assert result.pair_probability_max_abs_difference.eq(0).all()


def test_full_cluster_real_manifest():
    frame=pd.read_csv(ROOT/'code/results/external_cohort_v1_4/downloaded.csv')
    result=m.exclude_cluster(frame)
    assert (len(result),result.observation_id.nunique(),result.taxon_id.nunique())==(964,337,13)
    assert result[result.taxon_id==4328].observer_id.nunique()==9


def test_new_entrypoint_has_no_photo_downloader():
    import ast
    source=(ROOT/'code/scripts/run_external_cohort_v1_5.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    forbidden={'download','materialize','duplicate_audit'}
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and
                isinstance(n.func,ast.Attribute) and n.func.attr in forbidden]
