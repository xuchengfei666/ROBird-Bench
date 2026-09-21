import io
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from robird import external_cohort_v1_3 as mod
from robird.external_cohort_v1_1 import GateStop
from robird.io import atomic_write_csv
from robird.rsos_suite_v1 import finish,shuffle_membership,dataset


def relations():
    rows=[]
    for obs,taxon,label,photos in [(377824923,4328,5,[690953732,11,12]),
                                  (377824924,4381,9,[690953732,21,22]),
                                  (99,8000,31,[41,42])]:
        for p in photos:
            rows.append(dict(observation_id=obs,taxon_id=taxon,class_index=label,photo_id=p,
                observer_id=2132783 if obs!=99 else 77,url=f'https://static.inaturalist.org/photos/{p}/large.png',
                relation_id=f'{obs}:{p}',shared_scene=p==mod.SHARED_PHOTO,split='development_test'))
    return pd.DataFrame(rows)


def test_only_reviewed_shared_identity_allowed():
    frame=relations()
    mod.assert_relation_metadata(frame,[])
    with pytest.raises(GateStop):mod.assert_relation_metadata(pd.concat([frame,frame.iloc[:1]]),[])
    modified=frame.copy();modified.loc[modified.photo_id==21,'photo_id']=11
    with pytest.raises(GateStop):mod.assert_relation_metadata(modified,[])
    with pytest.raises(GateStop):mod.assert_relation_metadata(frame,[pd.DataFrame(dict(photo_id=[11]))])
    modified=frame.copy();modified.loc[modified.observation_id==377824924,'taxon_id']=4328
    with pytest.raises(GateStop):mod.assert_relation_metadata(modified,[])


def test_sensitivity_keeps_groups_and_original_records():
    original=relations();saved=original.copy(deep=True)
    result=mod.exclusion_sensitivity(original)
    assert len(result)==len(original)-2
    assert set(result.observation_id)==set(original.observation_id)
    assert result.groupby('observation_id').size().eq(2).all()
    pd.testing.assert_frame_equal(original,saved)


def test_unique_features_expand_without_overwriting_labels():
    frame=relations();unique=frame.drop_duplicates('photo_id')
    features=np.arange(len(unique)*4).reshape(len(unique),4)
    expanded=mod.expand_features(frame,unique,features)
    shared=frame.photo_id.eq(mod.SHARED_PHOTO).to_numpy()
    assert np.array_equal(expanded[shared][0],expanded[shared][1])
    data=dataset(frame,expanded,'development_test',1,20260819,'exhaustive')
    items=[data[i] for i in range(len(data)) if data[i]['photo_ids']==[mod.SHARED_PHOTO]]
    assert len(items)==2 and {x['label'] for x in items}=={5,9}
    assert np.array_equal(items[0]['features'].numpy(),items[1]['features'].numpy())


def test_quality_cache_rebinds_relation_not_label():
    cache=dict(photo_id=mod.SHARED_PHOTO,observation_id=377824923,taxon_id=4328,class_index=5,
               observer_id=2132783,laplacian_variance=.02)
    relation=relations()[relations().observation_id==377824924].iloc[0].to_dict()
    result=mod.relation_quality_row(cache,relation)
    assert result['observation_id']==377824924 and result['class_index']==9
    assert result['laplacian_variance']==.02 and cache['class_index']==5


def test_shared_duplicate_exception_never_exempts_history_or_other_hash():
    a=dict(photo_id=mod.SHARED_PHOTO,observation_id=377824923,sha256=mod.SHARED_HASH)
    b=dict(photo_id=mod.SHARED_PHOTO,observation_id=377824924,sha256=mod.SHARED_HASH)
    assert mod.accepted_shared_pair(a,b,'within_new_cohort')
    assert not mod.accepted_shared_pair(a,b,'history')
    assert not mod.accepted_shared_pair(a,dict(b,sha256='different'),'within_new_cohort')
    assert not mod.accepted_shared_pair(a,dict(b,observation_id=123),'within_new_cohort')


def test_duplicate_gate_retains_reviewed_pair_but_stops_others(tmp_path):
    data=tmp_path/'data';report=tmp_path/'report';data.mkdir();report.mkdir()
    reference=data/'reference_signatures.csv'
    atomic_write_csv(reference,pd.DataFrame([dict(local_path='old.png',photo_id=1,observation_id=1,
         sha256='historical',dhash='ffffffffffffffff')]))
    finish(data/'reference_signatures_done.json','test',[reference])
    frame=pd.DataFrame([dict(photo_id=mod.SHARED_PHOTO,observation_id=obs,sha256=mod.SHARED_HASH,
        local_path='review.jpg',dhash='0000000000000000') for obs in mod.SHARED_OBSERVATIONS])
    mod.duplicate_audit(frame,tmp_path,data,report,{},'test')
    result=json.loads((report/'duplicate_audit.json').read_text())
    assert result['reviewed_shared_pairs']==1 and result['unresolved_pairs']==0
    bad=pd.concat([frame,pd.DataFrame([dict(photo_id=9,observation_id=88,sha256='new',local_path='new.jpg',dhash='0000000000000000')])])
    with pytest.raises(GateStop):mod.duplicate_audit(bad,tmp_path,data,tmp_path/'badreport',{},'test')
    result=json.loads((tmp_path/'badreport/duplicate_audit.json').read_text())
    assert result['unresolved_pairs']==2


def test_download_once_per_image_and_preserve_all_relations(monkeypatch,tmp_path):
    image=tmp_path/'review.png';Image.new('RGB',(20,20),'white').save(image)
    monkeypatch.setattr(mod,'REVIEW',image)
    monkeypatch.setattr(mod,'SHARED_HASH',hashlib.sha256(image.read_bytes()).hexdigest())
    calls=[]
    class Response(io.BytesIO):pass
    content=image.read_bytes()
    def fetch(request,**kwargs):
        calls.append(request.full_url);response=Response(content);response.url=request.full_url;return response
    monkeypatch.setattr(mod.legacy,'urlopen',fetch)
    monkeypatch.setattr(mod.legacy.time,'sleep',lambda *_:None)
    frame=relations()
    result=mod.download(frame,tmp_path/'data',tmp_path/'report','test')
    assert len(result)==8 and result.photo_id.nunique()==7
    assert len(calls)==6  # one reviewed image reused, six other unique images fetched
    shared=result[result.photo_id==mod.SHARED_PHOTO]
    assert shared.local_path.nunique()==1 and shared.class_index.nunique()==2
    again=mod.download(frame,tmp_path/'data',tmp_path/'report','test')
    assert len(calls)==6 and len(again)==8


def test_shuffle_preserves_relation_pool_with_cross_taxon_shared_image():
    frame=relations();rows=[]
    for _,row in frame[frame.observation_id.isin(mod.SHARED_OBSERVATIONS)].iterrows():
        rows.append(row.to_dict())
        for shift in (1000,2000):
            new=row.to_dict();new['observation_id']+=shift;new['photo_id']+=shift
            rows.append(new)
    source=pd.DataFrame(rows)
    for seed in (2026090800,2026090801):
        shuffled,mapping=shuffle_membership(source,seed)
        assert len(shuffled)==len(source)
        assert not (mapping.slot_observation_id==mapping.source_observation_id).any()
        original=source.groupby(['taxon_id','photo_id']).size().sort_index()
        after=shuffled.groupby(['taxon_id','photo_id']).size().sort_index()
        pd.testing.assert_series_equal(original,after)
        assert (mapping.groupby('slot_observation_id').source_observation_id.nunique()>=2).all()
