"""Finalize already computed v1 results without changing or rerunning them."""
from pathlib import Path
import json
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/src'))
from robird.io import atomic_write_json,sha256_file
from robird.rsos_suite_v1 import finish,marker
from robird.external_cohort_v1_1 import GateStop,read_json
from robird.artifact_tree_v1 import verify_tree
from robird.external_completion_v1 import adjust_families
SOURCE=ROOT/'code/results/external_completion_v1'
REPORT=ROOT/'code/results/external_completion_v1_1'
PARENT=ROOT/'FROZEN_EXTERNAL_COMPLETION_V1.json'
FREEZE=ROOT/'FROZEN_EXTERNAL_COMPLETION_V1_1.json'
EXPECTED='d5ed9c2948e3cc54434678f42c8a0b729b4b00fb4e444f4af22dec72bbbed28b'


def validate_aggregates(source=SOURCE):
    individual=sorted(source.glob('*/runs/*/done.json'));mean=sorted(source.glob('*/seed_mean/*/done.json'))
    if len(individual)!=48 or len(mean)!=16:raise GateStop('Expected48 individual and16 seed-mean markers')
    for path in individual+mean:
        if read_json(path)['contract']!=EXPECTED:raise GateStop('Calculation contract mismatch')
        verify_tree(path)
    expected=pd.concat([pd.read_csv(p.parent/'metric_intervals.csv') for p in individual+mean],ignore_index=True)
    actual=pd.read_csv(source/'all_metric_intervals.csv')
    keys=['mode','model','training_policy','seed','cohort','taxon_id','budget','metric']
    for frame in (expected,actual):
        frame['seed']=frame.seed.astype(str);frame['taxon_id']=frame.taxon_id.astype(str)
        if frame[keys].duplicated().any():raise GateStop('Repeated interval identity')
    pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
        actual.sort_values(keys).reset_index(drop=True),check_exact=False,rtol=0,atol=1e-14)
    if len(actual)!=28800:raise GateStop('Unexpected interval matrix size')
    numeric=actual[['point','lower95','upper95']]
    if not np.isfinite(numeric.point).all():raise GateStop('Nonfinite point estimate')
    invalid=actual.status.eq('NOT_ESTIMABLE_LT2_OBSERVERS')
    if not (actual.loc[invalid,'observers']<2).all() or not actual.loc[invalid,['lower95','upper95']].isna().all().all():
        raise GateStop('Invalid rare-observer CI policy')
    valid=~invalid
    if not np.isfinite(actual.loc[valid,['lower95','upper95']]).all().all() or not (actual.loc[valid,'lower95']<=actual.loc[valid,'upper95']).all():
        raise GateStop('Invalid interval range')
    groupkeys=[k for k in keys if k!='seed']
    seeds=actual[actual.seed!='mean_of_three'];averaged=actual[actual.seed=='mean_of_three']
    if not seeds.groupby(groupkeys).seed.nunique().eq(3).all():raise GateStop('Seed coverage changed')
    check=seeds.groupby(groupkeys).point.mean().reset_index().merge(averaged[groupkeys+['point']],on=groupkeys,suffixes=('_expected','_actual'),validate='one_to_one')
    if len(check)!=len(averaged) or not np.allclose(check.point_expected,check.point_actual,atol=1e-10,rtol=0):
        raise GateStop('Seed metric mean mismatch')
    paired=read_json(source/'paired_budget_tests.json')
    if len(paired)!=64:raise GateStop('Incomplete paired tests')
    corrected=adjust_families(paired)
    for original,new in zip(paired,corrected):
        for key in ('observer_signflip_p','taxon_signflip_p'):
            for suffix in ('','_holm16','_holm64'):
                value=original[key+suffix]
                if not 0<=value<=1 or not np.isclose(value,new[key+suffix],atol=1e-14,rtol=0):
                    raise GateStop('Invalid multiplicity correction')
    return dict(statistics_complete=True,human_quality_complete=False,original_g6_pass=False,
        individual_runs=48,seed_mean_groups=16,interval_records=len(actual),paired_tests=64,
        unestimable_interval_records=int(invalid.sum()),empirical_degenerate_intervals=int(actual.status.eq('EMPIRICAL_DEGENERATE').sum()),
        source_outputs_unchanged=True,new_bootstrap_runs=0,new_model_runs=0),individual+mean


def main():
    if sha256_file(PARENT)!=EXPECTED:raise GateStop('Parent v1 freeze changed')
    seen={}
    for name,expected in read_json(PARENT)['files'].items():
        path=(ROOT/name).resolve()
        if path not in seen:seen[path]=sha256_file(path)
        if seen[path]!=expected:raise GateStop('Frozen input changed: '+str(path))
    result,markers=validate_aggregates()
    artifacts=[SOURCE/n for n in ('all_metric_intervals.csv','paired_budget_tests.json','paired_budget_tests.csv',
        'assumptions_and_completeness.json','resolution_failure_strata.csv')]+markers
    if not FREEZE.exists():
        extra=[PARENT,Path(__file__),ROOT/'EXTERNAL_COMPLETION_FINALIZER_V1_1.md',
            ROOT/'code/src/robird/artifact_tree_v1.py',ROOT/'code/tests/test_artifact_tree_v1.py',
            SOURCE/'queue_status.json',SOURCE/'host_status.json',*artifacts]
        atomic_write_json(FREEZE,dict(parent_contract=EXPECTED,files={str(p.resolve()):sha256_file(p) for p in extra},
            correction='Array JSON leaves supported; all artifact hashes still required',created_unix=time.time()),refuse_if_exists=True)
    else:
        for name,expected in read_json(FREEZE)['files'].items():
            if sha256_file(name)!=expected:raise GateStop('Finalizer input changed')
    contract=sha256_file(FREEZE)
    if not marker(REPORT/'statistics_done.json',contract):
        atomic_write_json(REPORT/'completion_audit.json',result,refuse_if_exists=True)
        finish(REPORT/'statistics_done.json',contract,[*artifacts,REPORT/'completion_audit.json'],**result)
    verify_tree(REPORT/'statistics_done.json')
    atomic_write_json(REPORT/'queue_status.json',dict(status='STATISTICS_COMPLETE_HUMAN_QUALITY_PENDING',
        finished_unix=time.time(),contract=contract,**result))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
