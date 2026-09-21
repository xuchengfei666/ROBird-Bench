"""CPU-only missing-reporting-item completion, with hash-bound resumable runs."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/src'))
import numpy as np
import pandas as pd
from robird import external_completion_v1 as m
from robird.io import atomic_write_json,atomic_write_csv,sha256_file
from robird.rsos_suite_v1 import marker,finish,atomic_binary
from robird.external_cohort_v1_1 import read_json,GateStop,verify_tree
SOURCE=ROOT/'code/results/external_cohort_v1_5'
SOURCE_DATA=Path('E:/Datasets/ROBird-Bench/external_cohort_v1_5')
REPORT=ROOT/'code/results/external_completion_v1'
DATA=Path('E:/Datasets/ROBird-Bench/external_completion_v1')
FREEZE=ROOT/'FROZEN_EXTERNAL_COMPLETION_V1.json'
CONFIG=ROOT/'code/configs/external_completion_v1.json'


def verify_files(files):
    seen={}
    for name,expected in files.items():
        path=(ROOT/name).resolve()
        if path not in seen:seen[path]=sha256_file(path)
        if seen[path]!=expected:raise GateStop('Frozen input changed: '+str(path))


def prepare():
    if FREEZE.exists():raise FileExistsError('Supplement already frozen')
    parent=ROOT/'FROZEN_EXTERNAL_COHORT_V1_5.json'
    if sha256_file(parent)!='90298d8ece98ec54c8318b3eeecc5e60b5911dce3ec714effb1d9d5b17bcf428':
        raise GateStop('v1.5 parent changed')
    inherited=read_json(parent)['files'];verify_files(inherited);verify_tree(SOURCE/'automatic_done.json')
    config=dict(parent_contract=sha256_file(parent),bootstrap_repeats=5000,signflip_repeats=9999,
                random_seed=20260909,modes=['full','sensitivity_no_cluster'],
                runs=read_json(ROOT/'code/configs/external_cohort_v1_5.json')['runs'],
                source_report=str(SOURCE),source_data=str(SOURCE_DATA),data_root=str(DATA),
                statistical_reporting_only=True,new_training=False,original_g6_pass=False,
                human_quality_complete=False)
    if CONFIG.exists():
        if read_json(CONFIG)!=config:raise GateStop('Partial config changed')
    else:atomic_write_json(CONFIG,config,refuse_if_exists=True)
    files=dict(inherited)
    extra=[parent,CONFIG,Path(__file__),ROOT/'EXTERNAL_COMPLETION_V1_PROTOCOL.md',
           ROOT/'code/src/robird/external_completion_v1.py',ROOT/'code/tests/test_external_completion_v1.py',
           ROOT/'code/scripts/quality_review_v1.py',ROOT/'code/assets/quality_review_v1.html',
           ROOT/'code/tests/test_quality_review_v1.py',SOURCE/'automatic_done.json',
           SOURCE/'reused_source_relations.csv',SOURCE/'sensitivity_relations.csv',SOURCE/'quality_proxies.csv',
           SOURCE/'quality_rater_A_template.csv',SOURCE/'quality_rater_B_template.csv',
           DATA/'human_quality/queue_private.json']
    for mode in config['modes']:
        p=SOURCE if mode=='full' else SOURCE/mode
        d=SOURCE_DATA if mode=='full' else SOURCE_DATA/mode
        for spec in config['runs']:
            extra.extend([p/'runs'/spec['run_id']/name for name in ('metrics.json','groups.csv','per_taxon.csv','nested.csv','eval_done.json')])
            extra.append(d/'runs'/spec['run_id']/'predictions.csv')
        extra += [p/'statistics_done.json',p/'seed_mean_groups.csv']
    for path in extra:files[str(path.resolve())]=sha256_file(path)
    atomic_write_json(FREEZE,dict(files=files,parent_sha256=sha256_file(parent),created_unix=time.time(),
        retrospective_reporting=True,no_new_training=True),refuse_if_exists=True)
    print(json.dumps(dict(status='PREPARED',contract=sha256_file(FREEZE),bound_files=len(files))),flush=True)


def run():
    import msvcrt
    REPORT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/'worker.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    contract=sha256_file(FREEZE);stage='verify_inputs'
    def status(state='RUNNING',**extra):
        v=dict(status=state,stage=stage,pid=os.getpid(),updated_unix=time.time(),contract=contract,**extra)
        atomic_write_json(REPORT/'queue_status.json',v);print(json.dumps(v),flush=True)
    try:
        status();verify_files(read_json(FREEZE)['files']);verify_tree(SOURCE/'automatic_done.json')
        if marker(REPORT/'statistics_done.json',contract):
            verify_tree(REPORT/'statistics_done.json');status('STATISTICS_COMPLETE_HUMAN_QUALITY_PENDING');return
        config=read_json(CONFIG);all_tables=[];all_pairs=[];artifacts=[];completed=0
        for mode in config['modes']:
            p=SOURCE if mode=='full' else SOURCE/mode;d=SOURCE_DATA if mode=='full' else SOURCE_DATA/mode
            manifest=pd.read_csv(SOURCE/('reused_source_relations.csv' if mode=='full' else 'sensitivity_relations.csv'))
            for model in ('mean_feature','probability_mlp','deepsets','set_transformer'):
                for policy in ('all','k1'):
                    packets=[]
                    for spec in sorted([s for s in config['runs'] if (s['model'],s['training_policy'])==(model,policy)],key=lambda s:s['seed']):
                        stage='metric_intervals';status(mode=mode,run_id=spec['run_id'],completed_runs=completed,total_runs=48)
                        directory=REPORT/mode/'runs'/spec['run_id'];done=directory/'done.json'
                        prediction=d/'runs'/spec['run_id']/'predictions.csv'
                        packet=m.components(pd.read_csv(prediction),manifest)
                        m.verify_frozen_points(packet,read_json(p/'runs'/spec['run_id']/'metrics.json'))
                        m.verify_per_taxon_points(packet,pd.read_csv(p/'runs'/spec['run_id']/'per_taxon.csv'))
                        packets.append(packet)
                        if not marker(done,contract):
                            table=m.interval_table(packet,dict(mode=mode,run_id=spec['run_id'],model=model,training_policy=policy,seed=spec['seed']),config['bootstrap_repeats'])
                            atomic_write_csv(directory/'metric_intervals.csv',table)
                            finish(done,contract,[directory/'metric_intervals.csv',prediction],point_estimates_match_frozen=True)
                        all_tables.append(pd.read_csv(directory/'metric_intervals.csv'));artifacts.append(done);completed+=1
                    identity=dict(mode=mode,model=model,training_policy=policy,seed='mean_of_three',run_id=f'{model}-{policy}-mean_of_three')
                    combined=m.combine_seeds(packets);directory=REPORT/mode/'seed_mean'/f'{model}-{policy}';done=directory/'done.json'
                    if not marker(done,contract):
                        table=m.interval_table(combined,identity,config['bootstrap_repeats'])
                        paired=m.paired_table(combined,identity,config['signflip_repeats'])
                        atomic_write_csv(directory/'metric_intervals.csv',table)
                        atomic_write_json(directory/'paired_tests_uncorrected.json',paired)
                        finish(done,contract,[directory/'metric_intervals.csv',directory/'paired_tests_uncorrected.json'],seed_estimand='mean_of_three_seed_metrics')
                    all_tables.append(pd.read_csv(directory/'metric_intervals.csv'));artifacts.append(done)
                    all_pairs.extend(read_json(directory/'paired_tests_uncorrected.json'))
        stage='resolution_failure_strata';status()
        resolution=[]
        for mode in config['modes']:
            path=SOURCE/('reused_source_relations.csv' if mode=='full' else 'sensitivity_relations.csv')
            frame=pd.read_csv(path);frame['pixel_area']=frame.width*frame.height
            frame['short_edge_px']=frame[['width','height']].min(axis=1)
            cov=frame.groupby('observation_id')[['pixel_area','short_edge_px']].mean().reset_index()
            source=SOURCE if mode=='full' else SOURCE/mode
            for spec in config['runs']:
                edge=pd.read_csv(source/'runs'/spec['run_id']/'nested.csv').merge(cov,on='observation_id',validate='many_to_one')
                for k,part in edge.groupby('budget_from'):
                    for field in ('pixel_area','short_edge_px'):
                        bins=pd.qcut(part[field],4,duplicates='drop') if part[field].nunique()>1 else pd.Series(['single_observed_value']*len(part),index=part.index)
                        for interval,piece in part.groupby(bins,observed=True):
                            resolution.append(dict(mode=mode,run_id=spec['run_id'],model=spec['model'],training_policy=spec['training_policy'],
                                seed=spec['seed'],budget_from=int(k),covariate=field,stratum=str(interval),groups=len(piece),
                                taxa=piece.label.nunique(),macro_regression=float(piece.groupby('label').regression.mean().mean()),
                                macro_net_gain=float(piece.groupby('label').net_gain.mean().mean()),causal_claim=False))
        atomic_write_csv(REPORT/'resolution_failure_strata.csv',pd.DataFrame(resolution))
        stage='aggregate_and_multiplicity';status()
        table=pd.concat(all_tables,ignore_index=True)
        adjusted=m.adjust_families(all_pairs)
        atomic_write_csv(REPORT/'all_metric_intervals.csv',table)
        atomic_write_json(REPORT/'paired_budget_tests.json',adjusted)
        flat=[]
        for row in adjusted:
            v={k:x for k,x in row.items() if not isinstance(x,dict)}
            for key in ('observer_interval','taxon_interval'):
                v[key+'_lower95'],v[key+'_upper95']=row[key]['interval95']
            flat.append(v)
        atomic_write_csv(REPORT/'paired_budget_tests.csv',pd.DataFrame(flat))
        diag=dict(retrospective_reporting=True,point_checks_passed=48,new_training_runs=0,new_inference_runs=0,
            bootstrap_repeats=5000,monte_carlo_signflips=9999,paired_comparisons=len(adjusted),
            interval_rows=len(table),unestimable_rows=int(table.status.eq('NOT_ESTIMABLE_LT2_OBSERVERS').sum()),
            empirical_degenerate_rows=int(table.status.eq('EMPIRICAL_DEGENERATE').sum()),
            max_observer_weight=max(r['max_cluster_weight'] for r in adjusted),
            independence_and_sign_symmetry='ASSUMPTIONS_NOT_PROVEN_BY_NORMALITY_TESTS',
            interval_scope='pointwise_not_simultaneous',uncertainty_does_not_remove_taxon_selection_bias=True,
            human_quality_complete=False,original_g6_pass=False)
        atomic_write_json(REPORT/'assumptions_and_completeness.json',diag)
        artifacts += [REPORT/n for n in ('all_metric_intervals.csv','paired_budget_tests.json','paired_budget_tests.csv','assumptions_and_completeness.json','resolution_failure_strata.csv')]
        for path in artifacts:
            if path.suffix=='.json':verify_tree(path)
        finish(REPORT/'statistics_done.json',contract,artifacts,statistics_complete=True,human_quality_complete=False)
        stage='human_annotation_boundary';status('STATISTICS_COMPLETE_HUMAN_QUALITY_PENDING',completed_runs=48)
    except Exception as exc:
        status('STOP_ENGINEERING_OR_INPUT_GATE',reason=str(exc),traceback=traceback.format_exc());raise
    finally:lock.close()


def host():
    import msvcrt
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/'host.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    attempt=REPORT/'host'/f'{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}';attempt.mkdir(parents=True)
    python=Path('C:/Users/Administrator/anaconda3/envs/pytorch1.0/python.exe')
    env=dict(os.environ);env['PYTHONUNBUFFERED']='1';env['PYTHONFAULTHANDLER']='1'
    with (attempt/'stdout.log').open('ab',buffering=0) as out,(attempt/'stderr.log').open('ab',buffering=0) as err:
        child=subprocess.Popen([str(python),'-X','faulthandler','-u',str(Path(__file__).resolve())],
            cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            while True:
                queue=read_json(REPORT/'queue_status.json') if (REPORT/'queue_status.json').exists() else {}
                atomic_write_json(REPORT/'host_status.json',dict(host_pid=os.getpid(),child_pid=child.pid,
                    status='CHILD_RUNNING',attempt=str(attempt),queue=queue,updated_unix=time.time()))
                try:code=child.wait(timeout=15);break
                except subprocess.TimeoutExpired:pass
            queue=read_json(REPORT/'queue_status.json') if (REPORT/'queue_status.json').exists() else {}
            value=dict(host_pid=os.getpid(),child_pid=child.pid,exit_code=code,queue=queue,
                status='CHILD_EXITED',attempt=str(attempt),finished_unix=time.time())
            atomic_write_json(attempt/'exit.json',value,refuse_if_exists=True);atomic_write_json(REPORT/'host_status.json',value)
        finally:lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group()
    group.add_argument('--prepare',action='store_true');group.add_argument('--host',action='store_true');args=ap.parse_args()
    if args.prepare:prepare()
    elif args.host:host()
    else:run()
