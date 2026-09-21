from pathlib import Path
import pytest
from robird.io import atomic_write_json,sha256_file
from robird.artifact_tree_v1 import verify_tree
from robird.external_cohort_v1_1 import GateStop


def test_json_array_leaf_still_hash_bound(tmp_path):
    leaf=tmp_path/'data.json';atomic_write_json(leaf,[dict(p=.5),dict(p=.8)])
    parent=tmp_path/'done.json';atomic_write_json(parent,dict(artifacts={str(leaf):sha256_file(leaf)}))
    verify_tree(parent)
    atomic_write_json(leaf,[dict(p=.4)])
    with pytest.raises(GateStop):verify_tree(parent)


def test_nested_array_and_object_mixed(tmp_path):
    leaf=tmp_path/'rows.json';atomic_write_json(leaf,[])
    mid=tmp_path/'mid.json';atomic_write_json(mid,dict(artifacts={str(leaf):sha256_file(leaf)}))
    top=tmp_path/'top.json';atomic_write_json(top,dict(artifacts={str(mid):sha256_file(mid)}))
    verify_tree(top)


def test_malformed_marker_fails(tmp_path):
    file=tmp_path/'invalid.json';atomic_write_json(file,dict(artifacts=['not a mapping']))
    with pytest.raises(GateStop):verify_tree(file)
