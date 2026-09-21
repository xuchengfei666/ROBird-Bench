import itertools
import json
import numpy as np
import pandas as pd
import pytest
from robird import external_completion_v1 as m
from robird.budget_metrics_v2_1 import validate_records,summarize_run,cluster_interval


def fixture_predictions():
    manifest=[];predictions=[]
    for obs,label,n in [(1,3,2),(2,3,5),(3,81,3),(4,81,5)]:
        photos=list(range(obs*10,obs*10+n))
        for photo in photos:manifest.append(dict(observation_id=obs,photo_id=photo,taxon_id=label+100,observer_id=1 if obs in (1,3) else obs,class_index=label))
        for k in range(1,n+1):
            for subset in itertools.combinations(photos,k):
                p=np.full(100,.01/98);p[label]=.65 if (sum(subset)+k)%2 else .25;p[(label+1)%100]=.99-p[label]
                predictions.append(dict(observation_id=obs,budget=k,label=label,prediction=int(p.argmax()),photo_ids=';'.join(map(str,subset)),probabilities_json=json.dumps(p.tolist())))
    return pd.DataFrame(predictions),pd.DataFrame(manifest)


def test_components_and_points_match_frozen_definitions():
    p,manifest=fixture_predictions();packet=m.components(p,manifest)
    summary,groups,_=summarize_run(validate_records(p,manifest,num_classes=100),repeats=30)
    m.verify_frozen_points(packet,summary)
    reference=groups.groupby(['taxon_id','budget']).agg(top1=('expected_accuracy','mean'),nll=('expected_nll','mean'),brier=('expected_brier','mean')).reset_index()
    m.verify_per_taxon_points(packet,reference)
    reference.loc[0,'nll']+=.1
    with pytest.raises(m.GateStop):m.verify_per_taxon_points(packet,reference)


def test_macro_cluster_ci_retains_legacy_estimand():
    p,manifest=fixture_predictions();meta,values=m.components(p,manifest)[1]
    rows=m.metric_intervals(meta,values,repeats=500,seed=20260909)
    expected=cluster_interval(meta,'expected_accuracy',repeats=500,seed=20260909)
    row=next(r for r in rows if r['metric']=='macro_accuracy')
    np.testing.assert_allclose([row['lower95'],row['upper95']],expected['interval95'])
    assert row['observers']==3


def test_ece_seed_mean_not_ensemble_and_recomputed_in_bootstrap():
    meta=pd.DataFrame(dict(label=[1,1],observer_id=[1,2]))
    values=np.zeros((2,2,19));values[0,:,4]=.2;values[1,:,4]=-.2
    assert m.point_metrics(meta,values)['ece']==pytest.approx(.2)
    assert m.point_metrics(meta,values.mean(0))['ece']==0
    rows=m.metric_intervals(meta,values,repeats=40)
    row=next(r for r in rows if r['metric']=='ece')
    assert row['point']==pytest.approx(.2) and row['lower95']==pytest.approx(.2)


def test_single_observer_interval_is_na():
    meta=pd.DataFrame(dict(label=[1,1],observer_id=[5,5]));values=np.zeros((2,19))
    rows=m.metric_intervals(meta,values,repeats=20)
    assert all(r['status']=='NOT_ESTIMABLE_LT2_OBSERVERS' and r['lower95'] is None for r in rows)


def test_exact_signflip_absolute_tail_includes_ties_and_zeros():
    assert m.observer_signflip([1.,1.])['p']==.5
    assert m.observer_signflip([1.,-1.])['p']==1
    assert m.observer_signflip([0.,0.])['p']==1
    assert m.observer_signflip([1.,2.,3.])['p']==.25


def test_monte_carlo_never_zero_and_reproducible():
    a=m.observer_signflip(np.ones(20),repeats=99,seed=1)
    b=m.observer_signflip(np.ones(20),repeats=99,seed=1)
    assert a==b and a['p']>=.01 and a['draws']==99


def test_three_seed_identity_alignment_and_paired_coverage():
    p,manifest=fixture_predictions();packet=m.components(p,manifest)
    combined=m.combine_seeds([packet,packet,packet])
    rows=m.paired_table(combined,dict(mode='full',model='test',training_policy='all'),repeats=19)
    assert len(rows)==4 and rows[0]['groups']==4 and rows[0]['observers']==3
    assert rows[0]['observers_spanning_taxa']==1
    assert rows[0]['delta_accuracy']==pytest.approx(rows[0]['observer_interval']['point'])


def test_holm_families_complete_no_selection():
    rows=[dict(mode=mode,training_policy=policy,model=model,before=before,after=after,
        observer_signflip_p=.01,taxon_signflip_p=.02) for mode in ('full','sensitivity_no_cluster')
        for policy in ('all','k1') for model in range(4) for before,after in m.TRANSITIONS]
    result=m.adjust_families(rows)
    assert len(result)==64 and result[0]['observer_signflip_p_holm16']==.16
    assert result[0]['observer_signflip_p_holm64']==.64
    assert 'observer_signflip_p_holm16' not in rows[0]
    with pytest.raises(m.GateStop):m.adjust_families(rows[:-1])
