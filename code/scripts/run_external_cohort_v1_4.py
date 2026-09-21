"""v1.4 continuation: accept only the fully reviewed three-photo shared scene."""
from __future__ import annotations
import importlib.util,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'code/src'))
from robird.io import atomic_write_json,sha256_file
from robird.external_cohort_v1_1 import read_json
import robird.external_cohort_v1_3 as m
spec=importlib.util.spec_from_file_location('runner_v13',ROOT/'code/scripts/run_external_cohort_v1_3.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.DATA=Path('E:/Datasets/ROBird-Bench/external_cohort_v1_4');runner.REPORT=ROOT/'code/results/external_cohort_v1_4';runner.CONFIG=ROOT/'code/configs/external_cohort_v1_4.json';runner.FREEZE=ROOT/'FROZEN_EXTERNAL_COHORT_V1_4.json'
os.environ['HF_HOME']=str(runner.DATA/'hf_cache');os.environ['TORCH_HOME']=str(runner.DATA/'model_cache')
CLUSTER={690953732,690959646,690959649,690960036,690960029};PAIR_OBS={377824923,377824924}
def accept(a,b,source):
    return source=='within_new_cohort' and int(a['photo_id']) in CLUSTER and int(b['photo_id']) in CLUSTER and {int(a['observation_id']),int(b['observation_id'])}==PAIR_OBS and a['sha256']==b['sha256']
m.accepted_shared_pair=accept
def prepare():
    if runner.CONFIG.exists() or runner.FREEZE.exists(): raise FileExistsError('v1.4 already prepared')
    parent=ROOT/'FROZEN_EXTERNAL_COHORT_V1_3.json';inherited=read_json(parent)['files'];runner.verify_files(inherited)
    config=read_json(ROOT/'code/configs/external_cohort_v1_3.json');config['data_root']=str(runner.DATA);config['shared_scene_cluster']=sorted(CLUSTER);config['shared_scene_observations']=sorted(PAIR_OBS)
    files=dict(inherited)
    extras=[parent,ROOT/'EXTERNAL_COHORT_SHARED_CLUSTER_V1_4.md',Path(__file__),ROOT/'code/src/robird/external_cohort_v1_3.py',ROOT/'code/tests/test_external_cohort_v1_4.py',ROOT/'code/results/external_shared_scene_review_v1/shared_group_review.json']
    for p in extras: files[str(p.resolve())]=sha256_file(p)
    atomic_write_json(runner.CONFIG,config,refuse_if_exists=True);files[str(runner.CONFIG)]=sha256_file(runner.CONFIG)
    atomic_write_json(runner.FREEZE,dict(files=files,parent_sha256=sha256_file(parent),created_unix=time.time(),change='accept reviewed three-photo shared-scene cluster only'),refuse_if_exists=True)
    print(json.dumps(dict(status='PREPARED_V1_4',files=len(files),contract=sha256_file(runner.FREEZE))),flush=True)
def main():
    if '--prepare' in sys.argv: prepare();return
    if '--preflight-only' in sys.argv:
        runner.preflight(read_json(runner.CONFIG));print(json.dumps(dict(status='PREFLIGHT_PASS',checkpoints=24)));return
    runner.main()
if __name__=='__main__':main()
