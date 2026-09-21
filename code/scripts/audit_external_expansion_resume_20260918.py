"""One-off recovery audit; makes evidence copies, never edits frozen experiment code."""
from pathlib import Path
import json
import time

import run_external_expansion_v2_1 as task
from robird.io import atomic_write_json, sha256_file
from robird.rsos_suite_v1 import atomic_binary


def main():
    host_lock=task.engine.acquire_lock('host.lock')
    worker_lock=task.engine.acquire_lock('worker.lock')
    try:
        contract=sha256_file(task.FREEZE)
        if contract!='6e5cecd4680aa8869cd4aa61d6788c9c1848baa567536f491d6dbde3f1e96112':
            raise task.m.GateStop('Unexpected v2.1 freeze digest')
        files=task.m.read_json(task.FREEZE)['files']
        task.m.verify_files(files)
        print(json.dumps(dict(stage='freeze_verified',files=len(files))),flush=True)
        config=task.m.read_json(task.REPORT/'resolved_config.json')
        data=Path(config['data_root'])
        taxa=sorted((data/'metadata').glob('taxon_*.json'))
        pages=sorted((data/'metadata/pages').rglob('*.done.json'))
        checked=dict(files)
        manifests={}
        def audit_marker(path):
            path=Path(path).resolve()
            value=task.m.read_json(path)
            if value['contract']!=contract:
                raise task.m.GateStop('Resume contract mismatch: '+str(path))
            for name,expected in value['artifacts'].items():
                key=str(Path(name).resolve())
                actual=checked.get(key)
                if actual is None:
                    actual=sha256_file(Path(name));checked[key]=actual
                if actual!=expected:
                    raise task.m.GateStop('Committed artifact drift: '+name)
            manifests[str(path)]=sha256_file(path)
            return value
        results=[audit_marker(path)['result'] for path in taxa]
        for path in pages:
            audit_marker(path)
        target=data/'metadata/pages/979753'
        completed_current=[dict(path=str(path),sha256=sha256_file(path),
                                params=task.m.read_json(path)['params']) for path in sorted(target.glob('*.done.json'))]
        report=task.REPORT/'recovery'/time.strftime('%Y%m%d_%H%M%S')
        if report.exists():
            raise FileExistsError(report)
        report.mkdir(parents=True)
        state=task.m.read_json(task.REPORT/'host_status.json')
        attempt=Path(state['attempt'])
        sources={'queue_before.json':task.REPORT/'queue_status.json',
                 'host_before.json':task.REPORT/'host_status.json',
                 'stdout_before.log':attempt/'stdout.log',
                 'stderr_before.log':attempt/'stderr.log',
                 'api_quota_before.json':data/'metadata/api_quota.json'}
        copies={}
        for name,source in sources.items():
            destination=report/name
            atomic_binary(destination,lambda f,s=source:f.write(s.read_bytes()))
            digest=sha256_file(source)
            if sha256_file(destination)!=digest:
                raise task.m.GateStop('Recovery evidence copy mismatch')
            copies[name]=dict(source=str(source),sha256=digest)
        snapshot=report/'committed_markers.json'
        atomic_write_json(snapshot,manifests,refuse_if_exists=True)
        audit=dict(status='PASS_SAFE_TO_RESUME_UNCHANGED_V2_1',created_at=task.m.utc_now(),
            contract=contract,frozen_files_verified=len(files),taxa_committed=len(taxa),
            live_page_markers_verified=len(pages),
            eligible8=sum(r['eligible8'] for r in results),eligible10=sum(r['eligible10'] for r in results),
            interrupted_taxon=979753,interrupted_taxon_committed_pages=completed_current,
            historical_attempt_exit_present=(attempt/'exit.json').exists(),evidence_copies=copies,
            committed_markers_sha256=sha256_file(snapshot),
            root_cause='UNDETERMINED_ABRUPT_HOST_AND_WORKER_DISAPPEARANCE',
            code_or_protocol_changes=False,original_g6_pass=False)
        atomic_write_json(report/'audit.json',audit,refuse_if_exists=True)
        print(json.dumps(dict(path=str(report/'audit.json'),**{k:v for k,v in audit.items()
            if k not in ('interrupted_taxon_committed_pages','evidence_copies')})),flush=True)
    finally:
        worker_lock.close();host_lock.close()


if __name__=='__main__':
    main()
