import importlib.util
from pathlib import Path
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

prepare=load('prepare_expansion_duplicate_review_v1')
finalize=load('finalize_expansion_duplicate_review_v1')
impact=load('analyze_expansion_duplicate_impact_v1')

def fixture(exact=False,same=False,source='history',decision='DISTINCT_HISTORY'):
    index=dict(pairs=[dict(pair_id=0,source=source,exact_bytes=exact,same_observation=same,sheet='test.jpg')])
    notes=dict(reviewer_type='AI_ASSISTED_VISUAL_REVIEW',independent_human_adjudication=False,
               outcome_blinded=True,pairs=[dict(pair_id=0,decision=decision,rationale='Visible scene differs')])
    return index,notes

def test_identity_normalization():
    assert prepare.identity('123.0')==prepare.identity('123')==123
    assert prepare.identity('') is None and prepare.identity(None) is None
    with pytest.raises(ValueError):prepare.identity('123.5')

def test_exact_cannot_be_false_positive():
    i,n=fixture(exact=True)
    with pytest.raises(finalize.GateStop):finalize.validate(i,n)

def test_human_provenance_cannot_be_faked():
    i,n=fixture();n['independent_human_adjudication']=True
    with pytest.raises(finalize.GateStop):finalize.validate(i,n)

def test_incomplete_or_duplicate_decisions_rejected():
    i,n=fixture();n['pairs']=[]
    with pytest.raises(finalize.GateStop):finalize.validate(i,n)
    i,n=fixture();n['pairs']=n['pairs']*2
    with pytest.raises(finalize.GateStop):finalize.validate(i,n)

def test_same_observation_redundancy_not_cross_false_positive():
    i,n=fixture(same=True,source='within_new_cohort',decision='DISTINCT_CROSS_OBSERVATION')
    with pytest.raises(finalize.GateStop):finalize.validate(i,n)
    n['pairs'][0]['decision']='WITHIN_OBSERVATION_REDUNDANCY'
    assert finalize.validate(i,n)[0]['decision']=='WITHIN_OBSERVATION_REDUNDANCY'

def test_exact_inside_observation_is_explicit():
    i,n=fixture(exact=True,same=True,source='within_new_cohort',decision='EXACT_WITHIN_OBSERVATION')
    assert finalize.validate(i,n)[0]['exact_bytes']

def test_unresolved_is_not_pass():
    i,n=fixture(decision='UNRESOLVED')
    assert finalize.validate(i,n)[0]['decision']=='UNRESOLVED'

def test_detail_evidence_hash_and_pair_binding(tmp_path):
    sheet=tmp_path/'sheet.jpg';sheet.write_bytes(b'sheet')
    panel=tmp_path/'panel.jpg';panel.write_bytes(b'panel')
    i={'sheets':{str(sheet):finalize.sha256_file(sheet)}}
    details={'panels':[dict(pair_id=0,path=str(panel),sha256=finalize.sha256_file(panel))]}
    rows=[dict(pair_id=0,sheet=str(sheet),inspected_evidence=str(panel))]
    finalize.verify_visual_evidence(i,details,rows)
    rows[0]['pair_id']=1
    with pytest.raises(finalize.GateStop):finalize.verify_visual_evidence(i,details,rows)
    rows[0]['pair_id']=0;panel.write_bytes(b'changed')
    with pytest.raises(finalize.GateStop):finalize.verify_visual_evidence(i,details,rows)

def test_duplicate_components_transitive_and_order_independent():
    assert impact.components([1,2,3,4],[(3,2),(2,1)])==[[1,2,3],[4]]
    assert impact.components([4,3,2,1],[(1,2),(2,3)])==[[1,2,3],[4]]
    with pytest.raises(impact.GateStop):impact.components([1,2],[(1,3)])

def test_impact_reads_csv_integer_decimal_style_ids():
    assert impact.identity('72973223.0')==72973223
    assert impact.identity('') is None

def test_support_counts_distinct_observers_not_groups():
    frame=impact.pd.DataFrame([dict(taxon_id=1,observation_id=o,observer_id=7) for o in range(8) for _ in range(2)])
    s=impact.summarize(frame,[1,2])
    assert s['observations']==8 and s['observers']==1 and s['taxa_ge8']==0
    assert s['support']=={'1':1,'2':0} and s['groups_ge_k']['2']==8
