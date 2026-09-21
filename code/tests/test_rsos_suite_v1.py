import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from robird.rsos_suite_v1 import (
    atomic_binary, finish, make_model, marker, shuffle_membership)
from robird.budget_metrics_v2_1 import validate_records


def test_probability_mlp_equals_deepsets_at_single_view():
    torch.manual_seed(19)
    ds = make_model("deepsets", 8).eval()
    mlp = make_model("probability_mlp", 8).eval()
    mlp.load_state_dict(ds.state_dict(), strict=True)
    x = torch.randn(3,1,8)
    mask = torch.ones(3,1,dtype=torch.bool)
    with torch.no_grad():
        torch.testing.assert_close(ds(x,mask).logits.softmax(-1),
                                   mlp(x,mask).logits.softmax(-1))


@pytest.mark.parametrize("name", ["mean_feature","probability_mlp","deepsets","set_transformer"])
def test_suite_models_padding_permutation_finite_gradient(name):
    torch.manual_seed(41)
    model = make_model(name, 8).eval()
    x = torch.randn(2,3,8)
    mask = torch.ones(2,3,dtype=torch.bool)
    p = model(x,mask).logits.softmax(-1)
    torch.testing.assert_close(p, model(x[:,[2,0,1]],mask).logits.softmax(-1), atol=1e-6, rtol=1e-5)
    xp = torch.cat([x,torch.randn(2,2,8)*99],dim=1)
    mp = torch.cat([mask,torch.zeros(2,2,dtype=torch.bool)],dim=1)
    torch.testing.assert_close(p, model(xp,mp).logits.softmax(-1), atol=1e-6, rtol=1e-5)
    loss = -p[:,0].log().mean()
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in model.parameters() if t.grad is not None)


def test_twenty_shuffles_preserve_pool_cardinality_and_source_identity():
    rows = []
    for label in (0,99):
        for j in range(4):
            obs = label*100+j
            for k in range(3):
                rows.append(dict(observation_id=obs, observer_id=obs+5000,
                                 taxon_id=label+100, class_index=label,
                                 photo_id=obs*10+k, split="development_test"))
    manifest = pd.DataFrame(rows)
    fingerprints=set()
    for seed in range(2026090800,2026090820):
        pseudo, mapping = shuffle_membership(manifest,seed)
        assert set(pseudo.photo_id) == set(manifest.photo_id)
        assert not pseudo.photo_id.duplicated().any()
        assert (mapping.source_observation_id != mapping.slot_observation_id).all()
        assert mapping.groupby("slot_observation_id").source_observation_id.nunique().min() >= 2
        assert pseudo.groupby("observation_id").size().eq(3).all()
        assert pseudo.observer_id.eq(-1).all()
        merged=mapping.merge(manifest[["photo_id","observer_id"]],on="photo_id")
        assert merged.source_observer_id.eq(merged.observer_id).all()
        fingerprints.add(tuple(pseudo.photo_id))
    assert len(fingerprints)>1


def test_done_hashes_detect_tampering(tmp_path):
    path=tmp_path/"a.npy"
    atomic_binary(path,lambda f:np.save(f,np.array([1,2])))
    finish(tmp_path/"done.json","contract",[path])
    assert marker(tmp_path/"done.json","contract")
    with pytest.raises(RuntimeError):
        marker(tmp_path/"done.json","wrong")
    atomic_binary(path,lambda f:np.save(f,np.array([2,3])))
    with pytest.raises(RuntimeError):
        marker(tmp_path/"done.json","contract")


def test_partial_taxon_cohort_keeps_fixed_output_dimension():
    import itertools
    rows=[]
    probs=np.zeros(100); probs[99]=1
    manifest=pd.DataFrame([dict(photo_id=i,observation_id=1,observer_id=-1,
                                taxon_id=1099,class_index=99) for i in [10,11]])
    for k in (1,2):
        for subset in itertools.combinations([10,11],k):
            rows.append(dict(observation_id=1,label=99,budget=k,
                             photo_ids=";".join(map(str,subset)),prediction=99,
                             probabilities_json=json.dumps(probs.tolist())))
    assert len(validate_records(pd.DataFrame(rows),manifest,num_classes=100)) == 1
