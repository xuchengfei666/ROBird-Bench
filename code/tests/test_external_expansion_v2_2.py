import copy
import itertools
import json
import numpy as np
import pandas as pd
import pytest
from robird import external_expansion_v2_2 as m

def source_frame(n=10):
    rows=[]
    for i in range(n):
        for j in range(2):
            p=100+2*i+j
            rows.append(dict(photo_id=p,observation_id=10+i,observer_id=1000+i,taxon_id=1,class_index=0,
                scientific_name='Test taxon',sha256=str(p).zfill(64),local_path=f'E:/test/{p}.jpg',
                license_code='cc-by',attribution='Synthetic author',original_url=f'https://static.inaturalist.org/photos/{p}/square.jpg',
                url=f'https://static.inaturalist.org/photos/{p}/large.jpg',strict10_taxon=True))
    return pd.DataFrame(rows)

def reviews(source,edges):
    pairs=[];notes=[]
    lookup=source.set_index('photo_id')
    for i,(a,b,decision) in enumerate(edges):
        ar,br=lookup.loc[a],lookup.loc[b]
        pairs.append(dict(pair_id=i,photo_id=a,reference_photo_id=b,observation_id=int(ar.observation_id),
            reference_observation_id=int(br.observation_id),path_a=ar.local_path,path_b=br.local_path,
            path_a_sha256=ar.sha256,path_b_sha256=br.sha256,source='within_new_cohort',
            same_observation=ar.observation_id==br.observation_id,exact_bytes=ar.sha256==br.sha256))
        notes.append(dict(pair_id=i,decision=decision,rationale='Synthetic frame identity test'))
    return dict(pairs=pairs),dict(pairs=notes,reviewer_type='AI_ASSISTED_VISUAL_REVIEW',
        independent_human_adjudication=False,outcome_blinded=True)

def validation_args(source):
    taxa=pd.DataFrame(dict(taxon_id=range(1,101),class_index=range(100)))
    excluded={c:set() for c in ('photo_id','observation_id','observer_id')}
    return source,taxa,excluded,dict(max_groups_per_observer_taxon=2,min_observers=8,sensitivity_min_observers=10)

def test_unique_frame_collapses_and_drops_under_two_without_mutating_source():
    source=source_frame();source.loc[1,'sha256']=source.loc[0,'sha256']
    old=source.copy(deep=True);i,n=reviews(source,[(101,100,'EXACT_WITHIN_OBSERVATION')])
    primary,sensitivity,mapping,_,affected,small=m.build_cohorts(source,i,n)
    assert m.counts(primary)==dict(photos=18,groups=9,observers=9,taxa=1,strict10_taxa=0)
    assert affected==small==[10] and len(sensitivity)==18
    assert not primary.strict10_taxon.any()
    assert int(mapping.loc[mapping.photo_id==101,'representative_photo_id'].iloc[0])==100
    pd.testing.assert_frame_equal(old,source)
    assert m.validate_cohort(primary,*validation_args(source))['groups']==9

def test_transitive_near_duplicate_component_and_minimum_id():
    source=source_frame(2)
    extra=source.iloc[0].copy();extra['photo_id']=99;extra['sha256']='9'*64;extra['local_path']='E:/test/99.jpg'
    source=pd.concat([source,pd.DataFrame([extra])],ignore_index=True)
    i,n=reviews(source,[(100,101,'NEAR_DUPLICATE_WITHIN_OBSERVATION'),(99,100,'NEAR_DUPLICATE_WITHIN_OBSERVATION')])
    _,_,mapping,_,_,_=m.build_cohorts(source,i,n)
    assert set(mapping[mapping.observation_id==10].representative_photo_id)=={99}

@pytest.mark.parametrize('decision',['UNRESOLVED','SUSPECT_CROSS_DUPLICATE','EXACT_HISTORY','EXACT_CROSS_OBSERVATION'])
def test_unsafe_or_unresolved_decisions_cannot_pass(decision):
    s=source_frame();i,n=reviews(s,[(100,101,decision)])
    with pytest.raises(m.GateStop):m.build_cohorts(s,i,n)

def test_cross_observation_cannot_be_relabeled_same_frame():
    s=source_frame();i,n=reviews(s,[(100,102,'NEAR_DUPLICATE_WITHIN_OBSERVATION')])
    with pytest.raises(m.GateStop):m.build_cohorts(s,i,n)

def test_missing_review_rejected():
    s=source_frame();i,n=reviews(s,[(100,101,'WITHIN_OBSERVATION_REDUNDANCY')]);n['pairs']=[]
    with pytest.raises(m.GateStop):m.build_cohorts(s,i,n)

def test_source_or_reference_hash_drift_rejected():
    s=source_frame();i,n=reviews(s,[(100,101,'WITHIN_OBSERVATION_REDUNDANCY')])
    s.loc[1,'sha256']='0'*64
    with pytest.raises(m.GateStop):m.build_cohorts(s,i,n)

@pytest.mark.parametrize('column,value',[('class_index',8),('attribution','changed'),('local_path','E:/other.jpg'),('observer_id',999)])
def test_source_fields_cannot_be_changed(column,value):
    source=source_frame();frame=source.copy();frame.loc[0,column]=value
    with pytest.raises(m.GateStop,match='Source fields'):m.validate_cohort(frame,*validation_args(source))

def test_history_overlap_and_thresholds():
    source=source_frame();args=list(validation_args(source));args[2]['observer_id'].add(1000)
    with pytest.raises(m.GateStop,match='History'):m.validate_cohort(source,*args)
    source=source_frame(7);frame=m.strict_support(source)
    with pytest.raises(m.GateStop,match='support'):m.validate_cohort(frame,*validation_args(source))
    source=source_frame(8);frame=m.strict_support(source)
    assert m.validate_cohort(frame,*validation_args(source))['strict10_taxa']==0

def test_sensitivity_cannot_drop_one_photo_of_an_observation():
    primary=source_frame();partial=primary.drop(index=0)
    with pytest.raises(m.GateStop,match='retained observation'):m.filter_prediction_manifest(primary,partial)

def test_whole_group_prediction_reuse_preserves_probabilities():
    primary=source_frame(2);sensitivity=primary[primary.observation_id==11]
    records=[]
    for obs,g in primary.groupby('observation_id'):
        for k in (1,2):
            for subset in itertools.combinations(g.photo_id,k):
                p=np.zeros(100);p[0]=.9;p[1]=.1
                records.append(dict(observation_id=obs,budget=k,photo_ids=';'.join(map(str,subset)),
                    label=0,prediction=0,probabilities_json=json.dumps(p.tolist())))
    predictions=pd.DataFrame(records)
    result=m.filtered_predictions(predictions,primary,sensitivity)
    pd.testing.assert_frame_equal(result,predictions[predictions.observation_id==11])
    with pytest.raises(ValueError):m.filtered_predictions(predictions.iloc[:-1],primary,sensitivity)

def test_model_blocked_without_new_gate(tmp_path):
    with pytest.raises(m.GateStop,match='missing unique-frame'):
        m.require_gate(tmp_path,tmp_path,{},'test',source_frame())

def test_expected_counts_not_adjusted_to_rescue_output():
    with pytest.raises(m.GateStop):m.check_counts({'photos':1},{'photos':2})

def test_integer_csv_float_style_and_fractional_rejection():
    assert m.integer('72973223.0')==72973223
    with pytest.raises(m.GateStop):m.integer('1.2')
