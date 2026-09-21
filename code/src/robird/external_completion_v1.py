"""Retrospective reporting only; consumes probabilities, never trains or infers."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from robird.budget_metrics_v2_1 import validate_records, group_statistics, cluster_interval
from robird.external_cohort_v1_1 import GateStop, holm, signflip, taxon_interval

METRICS=('macro_accuracy','micro_accuracy','top5','nll','brier','ece')
TRANSITIONS=((1,2),(2,3),(2,4),(2,5))


def components(predictions,manifest):
    """Arrays [groups, 4 scalar + 15 calibration signed-bin contributions]."""
    groups=validate_records(predictions,manifest,num_classes=100)
    budgets,_=group_statistics(groups)
    packets={}
    for k,part in budgets.groupby('budget',sort=True):
        part=part.sort_values('observation_id').reset_index(drop=True)
        rows=[]
        for obs in part.observation_id:
            g=groups[int(obs)]
            probs=np.stack([p for s,p in g['subsets'].items() if len(s)==int(k)])
            confidence=probs.max(1);correct=(probs.argmax(1)==g['label']).astype(float)
            bins=np.minimum((confidence*15).astype(int),14)
            residual=np.bincount(bins,weights=correct-confidence,minlength=15)/len(probs)
            y=g['label'];target=np.eye(100)[y]
            scalars=[correct.mean(),np.any(np.argsort(probs,axis=1)[:,-5:]==y,axis=1).mean(),
                     -np.log(np.clip(probs[:,y],1e-12,1)).mean(),np.square(probs-target).sum(1).mean()]
            rows.append(np.r_[scalars,residual])
        values=np.stack(rows)
        if not np.isfinite(values).all():raise GateStop('Nonfinite sufficient statistics')
        packets[int(k)]=(part,values)
    return packets


def point_metrics(meta,values):
    values=np.asarray(values,dtype=float)
    if values.ndim==2:values=values[None]
    class_index=[np.flatnonzero(meta.label.to_numpy()==v) for v in sorted(meta.label.unique())]
    means=values[:,:,:4].mean(1)
    macro=np.stack([values[:,idx,0].mean(1) for idx in class_index],axis=1).mean(1)
    ece=np.abs(values[:,:,4:].mean(1)).sum(1)
    # Mean of seed metrics, not argmax or calibration of ensembled probabilities.
    return dict(zip(METRICS,[float(macro.mean()),float(means[:,0].mean()),
        float(means[:,1].mean()),float(means[:,2].mean()),float(means[:,3].mean()),float(ece.mean())]))


def metric_intervals(meta,values,repeats=5000,seed=20260909):
    meta=meta.reset_index(drop=True)
    values=np.asarray(values,dtype=float)
    if values.ndim==2:values=values[None]
    if values.shape[1:]!=(len(meta),19):raise GateStop('Component/identity alignment mismatch')
    point=point_metrics(meta,values)
    ids,inverse=np.unique(meta.observer_id.to_numpy(),return_inverse=True)
    n=len(ids);base=dict(groups=len(meta),taxa=int(meta.label.nunique()),observers=n,repeats=repeats,
                        seed_count=values.shape[0],seed_estimand='mean_of_metrics_not_probability_ensemble')
    if n<2:
        return [dict(base,metric=k,point=v,lower95=None,upper95=None,status='NOT_ESTIMABLE_LT2_OBSERVERS',
                     interval_method='observer_cluster') for k,v in point.items()]
    sums=np.zeros((values.shape[0],n,19));sizes=np.bincount(inverse,minlength=n)
    for s,v in enumerate(values):np.add.at(sums[s],inverse,v)
    group_accuracy=values[:,:,0].mean(0)
    labels=meta.label.to_numpy();influence=np.zeros(len(meta))
    for label in np.unique(labels):
        idx=np.flatnonzero(labels==label)
        influence[idx]=(group_accuracy[idx]-group_accuracy[idx].mean())/(len(np.unique(labels))*len(idx))
    cluster_influence=np.bincount(inverse,weights=influence,minlength=n)
    if abs(cluster_influence.sum())>1e-10:raise GateStop('Uncentered macro influence')
    rng=np.random.default_rng(seed);samples=np.empty((repeats,6))
    for start in range(0,repeats,200):
        take=min(200,repeats-start)
        draws=rng.multinomial(n,np.ones(n)/n,size=take)
        denominator=draws@sizes
        mean=np.einsum('rc,scd->rsd',draws,sums,optimize=True)/denominator[:,None,None]
        samples[start:start+take,0]=point['macro_accuracy']+draws@cluster_influence
        samples[start:start+take,1:5]=mean[:,:,:4].mean(1)
        samples[start:start+take,5]=np.abs(mean[:,:,4:]).sum(2).mean(1)
    if not np.isfinite(samples).all():raise GateStop('Invalid bootstrap samples')
    output=[]
    for col,(name,value) in enumerate(point.items()):
        low,high=np.quantile(samples[:,col],[.025,.975])
        output.append(dict(base,metric=name,point=value,lower95=float(low),upper95=float(high),
            status='EMPIRICAL_DEGENERATE' if np.ptp(samples[:,col])<1e-14 else 'ESTIMATED',
            interval_method='linearized_observer_cluster_fixed_taxa' if col==0 else 'observer_cluster_ratio_percentile',
            interval_scope='pointwise_conditional_on_available_taxa_not_simultaneous'))
    return output


def verify_frozen_points(packets,metrics):
    mapping={'macro_accuracy':'macro_expected_accuracy','micro_accuracy':'micro_expected_accuracy',
        'top5':'group_equal_expected_top5','nll':'group_equal_expected_nll',
        'brier':'group_equal_expected_brier','ece':'group_equal_subset_ece'}
    for cohort in ('eligible_per_budget','fixed_five_photo_cohort'):
        for k,(meta,values) in packets.items():
            mask=np.ones(len(meta),dtype=bool) if cohort=='eligible_per_budget' else meta.n_photos.eq(5).to_numpy()
            if not mask.any():continue
            point=point_metrics(meta.loc[mask],values[mask])
            frozen=metrics[cohort][str(k)]
            for key,value in point.items():
                if not np.isclose(value,frozen[mapping[key]],atol=1e-10,rtol=0):
                    raise GateStop(f'Point drift: {cohort} k{k} {key}')


def verify_per_taxon_points(packets,reference):
    for k,(meta,values) in packets.items():
        for taxon in meta.taxon_id.unique():
            mask=meta.taxon_id.eq(taxon).to_numpy()
            old=reference[(reference.budget==k)&(reference.taxon_id==taxon)]
            if len(old)!=1:raise GateStop('Per-taxon frozen coverage mismatch')
            point=point_metrics(meta.loc[mask],values[mask])
            for key,col in [('macro_accuracy','top1'),('nll','nll'),('brier','brier')]:
                if not np.isclose(point[key],old.iloc[0][col],atol=1e-10,rtol=0):
                    raise GateStop('Per-taxon point estimate drift')


def interval_table(packets,identity,repeats=5000):
    result=[]
    for k,(meta,values) in packets.items():
        if values.ndim==2:values=values[None]
        masks=[('eligible_per_budget','ALL',np.ones(len(meta),dtype=bool)),
               ('fixed_five_photo_cohort','ALL',meta.n_photos.eq(5).to_numpy())]
        masks += [('per_taxon',int(t),meta.taxon_id.eq(t).to_numpy()) for t in sorted(meta.taxon_id.unique())]
        for cohort,taxon,mask in masks:
            if not mask.any():continue
            rows=metric_intervals(meta.loc[mask],values[:,mask],repeats)
            for row in rows:row.update(identity,cohort=cohort,taxon_id=taxon,budget=k)
            result.extend(rows)
    return pd.DataFrame(result)


def combine_seeds(packets):
    result={}
    if len(packets)!=3:raise GateStop('Exactly three fixed seeds required')
    if any(set(p)!=set(packets[0]) for p in packets):raise GateStop('Budget mismatch across seeds')
    keys=['observation_id','observer_id','label','taxon_id','n_photos','budget']
    for k in packets[0]:
        meta=packets[0][k][0]
        for packet in packets[1:]:pd.testing.assert_frame_equal(meta[keys],packet[k][0][keys])
        result[k]=(meta,np.stack([p[k][1] for p in packets]))
    return result


def observer_signflip(contributions,repeats=9999,seed=20260909):
    x=np.asarray(contributions,dtype=float)
    if not len(x) or not np.isfinite(x).all():raise GateStop('Invalid observer contributions')
    statistic=float(x.sum());cut=abs(statistic)-1e-12
    if len(x)<=13:
        signs=np.array(list(itertools.product((-1.,1.),repeat=len(x))))
        extreme=int((np.abs(signs@x)>=cut).sum())
        return dict(statistic=statistic,p=extreme/len(signs),method='exact_observer_signflip_absolute_tail',
                    draws=len(signs),extreme=extreme,monte_carlo_se=0.)
    rng=np.random.default_rng(seed);extreme=0
    for start in range(0,repeats,500):
        signs=2*rng.integers(0,2,size=(min(500,repeats-start),len(x)))-1
        extreme+=int((np.abs(signs@x)>=cut).sum())
    p=(1+extreme)/(1+repeats)
    return dict(statistic=statistic,p=p,method='monte_carlo_observer_signflip_absolute_tail_plus_one',
                draws=repeats,extreme=extreme,monte_carlo_se=float(np.sqrt(p*(1-p)/(repeats+1))))


def paired_table(combined,identity,repeats=9999):
    records=[]
    for before,after in TRANSITIONS:
        ma,va=combined[before];mb,vb=combined[after]
        a=ma[['observation_id','observer_id','label','taxon_id']].copy();a['before_accuracy']=va[:,:,0].mean(0)
        b=mb[['observation_id']].copy();b['after_accuracy']=vb[:,:,0].mean(0)
        pair=a.merge(b,on='observation_id',validate='one_to_one')
        pair['delta']=pair.after_accuracy-pair.before_accuracy
        count=pair.groupby('label').delta.transform('size');taxa=pair.label.nunique()
        weighted=pair.delta/(taxa*count)
        contributions=weighted.groupby(pair.observer_id).sum()
        test=observer_signflip(contributions.to_numpy(),repeats)
        weights=(1/(taxa*count)).groupby(pair.observer_id).sum()
        record=dict(identity,before=before,after=after,groups=len(pair),taxa=taxa,observers=len(contributions),
            delta_accuracy=test['statistic'],delta_percentage_points=100*test['statistic'],
            observer_signflip_p=test['p'],test_method=test['method'],draws=test['draws'],
            monte_carlo_se=test['monte_carlo_se'],taxon_signflip_p=signflip(pair,'delta'),
            observer_interval=cluster_interval(pair,'delta',5000),taxon_interval=taxon_interval(pair,'delta'),
            observers_spanning_taxa=int((pair.groupby('observer_id').label.nunique()>1).sum()),
            max_cluster_weight=float(weights.max()),max_cluster_groups=int(pair.groupby('observer_id').size().max()),
            cluster_sign_exchangeability='ASSUMED_NOT_EMPIRICALLY_VERIFIABLE',
            independent_observers='ASSUMED_SOURCE_IDENTITIES_ARE_NOT_PROOF',
            missing_or_outlier_removal=False,retrospective_reporting=True)
        records.append(record)
    return records


def adjust_families(records):
    result=[dict(r) for r in records]
    if len(result)!=64:raise GateStop('Expected all64 budget tests')
    families=sorted({(r['mode'],r['training_policy']) for r in result})
    for mode,policy in families:
        idx=[i for i,r in enumerate(result) if (r['mode'],r['training_policy'])==(mode,policy)]
        if len(idx)!=16:raise GateStop('Incomplete16-test family')
        for field in ('observer_signflip_p','taxon_signflip_p'):
            for i,p in zip(idx,holm([result[i][field] for i in idx])):result[i][field+'_holm16']=p
        for i in idx:result[i]['family']=f'{mode}/{policy}/four_models_x_four_transitions'
    for field in ('observer_signflip_p','taxon_signflip_p'):
        for record,p in zip(result,holm([r[field] for r in result])):record[field+'_holm64']=p
    return result
