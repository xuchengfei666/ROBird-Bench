"""Hash verification for marker objects with arbitrary valid JSON data leaves."""
from pathlib import Path
import json
from robird.io import sha256_file
from robird.external_cohort_v1_1 import GateStop


def verify_tree(path,seen=None):
    path=Path(path).resolve();seen=set() if seen is None else seen
    if path in seen:return
    seen.add(path)
    value=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value,dict) or 'artifacts' not in value:return
    artifacts=value['artifacts']
    if not isinstance(artifacts,dict):raise GateStop('Malformed artifact mapping: '+str(path))
    for name,expected in artifacts.items():
        if not isinstance(name,str) or not isinstance(expected,str) or len(expected)!=64:
            raise GateStop('Malformed hash entry: '+str(path))
        target=Path(name)
        if sha256_file(target)!=expected:raise GateStop('Input artifact changed: '+str(target))
        if target.suffix=='.json':verify_tree(target,seen)
