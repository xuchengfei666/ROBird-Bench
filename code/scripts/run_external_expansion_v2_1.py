"""Isolated compatibility continuation of v2; science and legacy engine are frozen."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import time
import sys
import traceback

import run_external_expansion_v2 as engine
from robird import external_expansion_v2 as m
from robird.external_expansion_v2_1 import canonical_taxa
from robird.io import atomic_write_csv, atomic_write_json, sha256_file
import pandas as pd

ROOT=engine.ROOT
REPORT=ROOT/'code/results/external_expansion_v2_1'
CONFIG=ROOT/'code/configs/external_expansion_v2_1.json'
FREEZE=ROOT/'FROZEN_EXTERNAL_EXPANSION_V2_1.json'
PARENT=ROOT/'FROZEN_EXTERNAL_EXPANSION_V2.json'
PARENT_HASH='d4161f7d18dab6426f6a92a09091d1e16e01bf3881df564491dfd04cd11fea1e'
# Explicit runtime dependency injection, not modification of the frozen v2 source.
engine.REPORT,engine.CONFIG,engine.FREEZE=REPORT,CONFIG,FREEZE


def prepare():
    if FREEZE.exists():
        raise FileExistsError('v2.1 already frozen')
    if sha256_file(PARENT)!=PARENT_HASH:
        raise m.GateStop('Parent v2 freeze changed')
    files=dict(m.read_json(PARENT)['files']);m.verify_files(files)
    config=engine.resolved_config()
    previous=m.read_json(ROOT/'code/configs/external_expansion_v2.json')
    allowed={'version','data_root','taxon_manifest','source_taxon_manifest'}
    authored=m.read_json(CONFIG)
    if {k:v for k,v in authored.items() if k not in allowed}!={k:v for k,v in previous.items() if k not in allowed}:
        raise m.GateStop('Scientific config changed in display-name correction')
    source=pd.read_csv(ROOT/config['source_taxon_manifest'])
    table=canonical_taxa(source)
    table_path=ROOT/config['taxon_manifest']
    if table_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(table_path),table)
    else:
        atomic_write_csv(table_path,table,refuse_if_exists=True)
    output=REPORT/'resolved_config.json'
    if output.exists():
        if m.read_json(output)!=config:
            raise m.GateStop('Partial prepare config changed')
    else:
        atomic_write_json(output,config,refuse_if_exists=True)
    paths=[PARENT,CONFIG,Path(__file__),table_path,output,
        ROOT/'EXTERNAL_EXPANSION_V2_1_ENGINEERING_CONTINUATION.md',
        ROOT/'code/src/robird/external_expansion_v2_1.py',
        ROOT/'code/tests/test_external_expansion_v2_1.py',
        ROOT/'code/results/external_expansion_v2/queue_status.json',
        ROOT/'code/results/external_expansion_v2/host_status.json']
    for path in paths:
        files[str(path.resolve())]=sha256_file(path)
    audit=dict(original_taxa=int(source.taxon_id.nunique()),original_classes=int(source.class_index.nunique()),
               display_name_distinct_rows=len(source[['taxon_id','class_index','scientific_name']].drop_duplicates()),
               canonical_rows=len(table),scientific_change=False,parent_sha256=PARENT_HASH)
    atomic_write_json(REPORT/'correction_audit.json',audit)
    files[str((REPORT/'correction_audit.json').resolve())]=sha256_file(REPORT/'correction_audit.json')
    atomic_write_json(FREEZE,dict(version='external_expansion_v2_1',created_at=m.utc_now(),files=files,
        parent_sha256=PARENT_HASH,correction='nonempty_display_names_only',original_g6_pass=False),refuse_if_exists=True)
    print(dict(status='FROZEN_V2_1',contract=sha256_file(FREEZE),files=len(files),**audit),flush=True)


def host():
    lock=engine.acquire_lock('host.lock')
    check=engine.acquire_lock('worker.lock');check.close()
    try:
        for restart in range(3):
            attempt=REPORT/'host'/f'{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}_{restart}'
            attempt.mkdir(parents=True,exist_ok=False)
            env=dict(os.environ);env.update(PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
            with (attempt/'stdout.log').open('ab',buffering=0) as out,(attempt/'stderr.log').open('ab',buffering=0) as err:
                child=subprocess.Popen([str(engine.PYTHON),'-X','faulthandler','-u',str(Path(__file__).resolve())],
                    cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                state=dict(status='CHILD_RUNNING',host_pid=os.getpid(),child_pid=child.pid,
                           attempt=str(attempt),restart=restart,started_at=m.utc_now())
                atomic_write_json(REPORT/'host_status.json',state)
                code=child.wait()
            queue=m.read_json(REPORT/'queue_status.json') if (REPORT/'queue_status.json').exists() else {}
            state.update(status='CHILD_EXITED',exit_code=code,queue=queue,finished_at=m.utc_now())
            atomic_write_json(attempt/'exit.json',state,refuse_if_exists=True)
            atomic_write_json(REPORT/'host_status.json',state)
            if code==0 or code==2 or queue.get('status')=='STOP_SCIENTIFIC_OR_INPUT_GATE':
                return
            if restart<2:
                time.sleep(60)
    except BaseException as exc:
        atomic_write_json(REPORT/'host_exception.json',dict(error=str(exc),traceback=traceback.format_exc(),at=m.utc_now()))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group()
    group.add_argument('--prepare',action='store_true');group.add_argument('--host',action='store_true')
    group.add_argument('--preflight-only',action='store_true');args=ap.parse_args()
    if args.prepare:
        prepare()
    elif args.host:
        host()
    else:
        sys.exit(engine.run(args.preflight_only))
