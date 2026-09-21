"""Process-lifetime diagnostics only; no changes to frozen scientific workflow."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'code/results/external_cohort_v1_3'
DATA=Path('E:/Datasets/ROBird-Bench/external_cohort_v1_3')
PYTHON=Path('C:/Users/Administrator/anaconda3/envs/pytorch1.0/python.exe')
FREEZE=ROOT/'FROZEN_EXTERNAL_COHORT_V1_3.json'
EXPECTED='3246db9379e30addd1a9a5adde21511c8382885f1a08beaf4e175fc8ac5bf36b'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.partial')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def validate_ledgers(directory,contract):
    verified=[]
    for path in sorted(Path(directory).glob('*.json')):
        value=read(path)
        if value['contract']!=contract:raise RuntimeError('Ledger contract changed: '+str(path))
        if not value.get('artifacts'):raise RuntimeError('Empty committed ledger')
        for file,digest in value['artifacts'].items():
            if sha(file)!=digest:raise RuntimeError('Image hash changed: '+file)
        verified.append(dict(photo_id=int(path.stem),ledger_sha256=sha(path),
                             image_sha256=value['details']['sha256']))
    return verified


def classify_exit(code,decision):
    status=decision.get('status','')
    if status.startswith('STOP_'):return 'CHILD_RECORDED_STOP'
    if code==0 and status=='AUTOMATIC_COMPLETE_HUMAN_QUALITY_ANNOTATION_PENDING':
        return 'AUTOMATIC_DONE_HUMAN_ANNOTATIONS_PENDING'
    return 'UNEXPECTED_CHILD_EXIT'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args()
    if sha(FREEZE)!=EXPECTED:raise RuntimeError('Frozen contract changed')
    stamp=time.strftime('%Y%m%d_%H%M%S')
    attempt=REPORT/'recovery_host_v1'/f'{stamp}_{os.getpid()}'
    attempt.mkdir(parents=True,exist_ok=False)
    for path in [REPORT/'queue_status.json',*REPORT.glob('queue_*.log')]:
        if path.exists():
            target=attempt/path.name
            with target.open('xb') as f:f.write(path.read_bytes())
    ledgers=validate_ledgers(DATA/'download_ledger',EXPECTED)
    write(attempt/'ledger_verification.json',dict(contract=EXPECTED,count=len(ledgers),ledgers=ledgers))
    if args.validate_only:
        print(json.dumps(dict(status='LEDGER_HASHES_PASS',count=len(ledgers),attempt=str(attempt))))
        return
    import msvcrt
    lock=(REPORT/'recovery_host.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    # Also fail closed if an existing worker already owns its scientific lock.
    check=(REPORT/'worker.lock').open('a+b')
    check.seek(0);msvcrt.locking(check.fileno(),msvcrt.LK_NBLCK,1);check.close()
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'code/src')
    env['PYTHONFAULTHANDLER']='1';env['PYTHONUNBUFFERED']='1'
    child=None
    def state(status,**extra):
        write(REPORT/'host_status.json',dict(status=status,host_pid=os.getpid(),
            child_pid=child.pid if child else None,contract=EXPECTED,attempt=str(attempt),
            initial_committed=len(ledgers),updated_unix=time.time(),**extra))
    try:
        with (attempt/'worker.stdout.log').open('ab',buffering=0) as out,(attempt/'worker.stderr.log').open('ab',buffering=0) as err:
            child=subprocess.Popen([str(PYTHON),'-X','faulthandler','-u',
                str(ROOT/'code/scripts/run_external_cohort_v1_3.py')],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            state('CHILD_RUNNING',committed_images=len(ledgers))
            while True:
                try:
                    code=child.wait(timeout=15);break
                except subprocess.TimeoutExpired:
                    paths=list((DATA/'download_ledger').glob('*.json'))
                    state('CHILD_RUNNING',committed_images=len(paths),
                          last_ledger_unix=max((p.stat().st_mtime for p in paths),default=None))
            decision=read(REPORT/'queue_status.json')
            record=dict(exit_code=code,exit_code_hex=f'0x{code & 0xffffffff:08X}',
                result=classify_exit(code,decision),decision=decision,finished_unix=time.time())
            write(attempt/'exit.json',record);state(record['result'],**record)
    except BaseException as exc:
        state('HOST_EXCEPTION',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:lock.close()


if __name__=='__main__':main()
