import io
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from robird.external_cohort_v1_1 import (GateStop,large_url,assert_metadata,
    image_signature,photo_proxies,detector_proxies,paired_budgets,holm,signflip,
    download,verify_tree,statistics,MODELS,SEEDS)
from robird.budget_metrics_v2_1 import group_statistics
from robird.io import atomic_write_json,sha256_file


def observations():
    return pd.DataFrame([dict(observation_id=i,observer_id=10+i//2,taxon_id=100+i,
        class_index=y,photo_id=i*2+j,license_code='cc-by')
        for i,y in enumerate((2,7,91)) for j in (1,2)])


@pytest.mark.parametrize('suffix',['jpg','jpeg',''])
def test_large_url_known_extension(suffix):
    assert large_url('https://inaturalist-open-data.s3.amazonaws.com/photos/5/square.'+suffix).endswith('/large.'+suffix)


@pytest.mark.parametrize('url',['https://evil.example/photos/1/square.jpg',
    'http://static.inaturalist.org/photos/1/square.jpg','https://static.inaturalist.org/photos/1/foo.jpg'])
def test_large_url_rejects_foreign_or_malformed(url):
    with pytest.raises(GateStop): large_url(url)


def test_identity_gate_including_sparse_labels_and_repeated_observers():
    frame=observations()
    assert_metadata(frame,[pd.DataFrame(dict(observer_id=[999]))])
    with pytest.raises(GateStop): assert_metadata(frame,[pd.DataFrame(dict(observer_id=[10]))])
    with pytest.raises(GateStop): assert_metadata(pd.concat([frame,frame.iloc[:1]]),[])


def test_matched_budget_not_changing_cohort():
    frame=pd.DataFrame([dict(observation_id=1,label=0,observer_id=1,budget=2,expected_accuracy=0.),
        dict(observation_id=2,label=0,observer_id=2,budget=2,expected_accuracy=1.),
        dict(observation_id=1,label=0,observer_id=1,budget=3,expected_accuracy=.25)])
    result=paired_budgets(frame,2,3)
    assert len(result)==1 and result.delta.iloc[0]==.25


def test_detector_missing_is_not_zero_area():
    value=detector_proxies(np.empty((0,4)),np.array([]),np.array([]),100,100,np.zeros((256,256)))
    assert value['bbox_area_fraction'] is None
    value=detector_proxies(np.array([[0.,0.,50.,50.]]),np.array([.9]),np.array([16]),100,100,np.zeros((256,256)))
    assert value['bbox_area_fraction']==.25 and value['bird_detection_count']==1


def test_quality_proxies_not_motion_or_occlusion_labels():
    result=photo_proxies(Image.new('RGB',(32,32),'white'))
    assert result['bright_fraction']==1 and result['laplacian_variance']==0
    assert not any('motion_blur' in k or 'occlusion' in k for k in result)


def test_image_hash_and_leading_zero_dhash(tmp_path):
    p=tmp_path/'test.png';Image.new('RGB',(20,10),'black').save(p)
    sig=image_signature(p)
    assert sig['dhash']=='0000000000000000' and len(sig['sha256'])==64


def test_download_ledger_resume_never_requeries(monkeypatch,tmp_path):
    from robird import external_cohort_v1_1 as mod
    content=io.BytesIO();Image.new('RGB',(20,20),'white').save(content,format='PNG')
    url='https://static.inaturalist.org/photos/1/large.png'
    class Response(io.BytesIO):
        pass
    def response(*args,**kwargs):
        value=Response(content.getvalue());value.url=url;return value
    monkeypatch.setattr(mod,'urlopen',response);monkeypatch.setattr(mod.time,'sleep',lambda *_:None)
    frame=pd.DataFrame([dict(photo_id=1,taxon_id=100,observation_id=9,observer_id=2,url=url)])
    result=download(frame,tmp_path/'data',tmp_path/'report','test')
    def fail(*a,**k):raise AssertionError('Resume re-downloaded')
    monkeypatch.setattr(mod,'urlopen',fail)
    restored=download(frame,tmp_path/'data',tmp_path/'report','test')
    assert restored.dhash.iloc[0]=='0000000000000000'
    assert result.sha256.iloc[0]==restored.sha256.iloc[0]


def test_recursive_marker_verification(tmp_path):
    leaf=tmp_path/'leaf.json';atomic_write_json(leaf,dict(data=2))
    branch=tmp_path/'branch.json';atomic_write_json(branch,dict(artifacts={str(leaf):sha256_file(leaf)}))
    top=tmp_path/'top.json';atomic_write_json(top,dict(artifacts={str(branch):sha256_file(branch)}))
    verify_tree(top)
    atomic_write_json(leaf,dict(data=3))
    with pytest.raises(GateStop): verify_tree(top)


def test_holm_and_signflip():
    assert np.allclose(holm([.01,.04,.03]),[.03,.06,.06])
    frame=pd.DataFrame(dict(label=[0,1,2],delta=[1.,1.,1.]))
    assert signflip(frame,'delta')==.25


def test_full_statistics_smoke_separates_policies(tmp_path):
    root=tmp_path/'root';report=tmp_path/'report';runs=[]
    for model in MODELS:
        for policy in ('all','k1'):
            for seed in SEEDS:
                run_id=f'dinov2-{model}-{policy}-seed{seed}'
                spec=dict(run_id=run_id,model=model,training_policy=policy,seed=seed)
                runs.append(spec)
                groups={}
                for i,y in enumerate((2,7,91)):
                    subsets={}
                    for k in (1,2):
                        for subset in itertools.combinations((2*i+1,2*i+2),k):
                            p=np.zeros(100);p[y if policy=='all' else (y+1)%100]=1
                            subsets[subset]=p
                    groups[i]=dict(photos=(2*i+1,2*i+2),label=y,observer_id=10+i//2,taxon_id=100+i,subsets=subsets)
                b,e=group_statistics(groups)
                destination=report/'runs'/run_id;destination.mkdir(parents=True)
                b.to_csv(destination/'groups.csv',index=False);e.to_csv(destination/'nested.csv',index=False)
                old=root/'code/results/rsos_serial_v1/runs'/run_id;old.mkdir(parents=True)
                b.to_csv(old/'groups.csv',index=False)
                atomic_write_json(old/'metrics.json',dict(spec=dict(backbone='dinov2',model=model,
                    budget=None if policy=='all' else 1,seed=seed),nested_transitions={}))
    statistics(root,report,dict(runs=runs),'test')
    result=pd.read_csv(report/'seed_summary.csv')
    assert result.loc[result.training_policy=='all','mean'].eq(1).all()
    assert result.loc[result.training_policy=='k1','mean'].eq(0).all()
    tests=json.loads((report/'statistical_tests.json').read_text())
    assert len(tests['model_contrasts'])==12
    assert all(x['groups']==0 for x in tests['paired_budget_tests'] if x['after']>2)
    # Completed output hashes are checked on resume.
    statistics(root,report,dict(runs=runs),'test')
