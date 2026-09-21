import numpy as np
import pandas as pd
import pytest
from robird.external_cohort_v1_1 import GateStop
from robird.external_cohort_v1_2 import canonical_taxon_map


def fixture_frame():
    return pd.DataFrame(dict(taxon_id=[11901]*4+[8021]*2,
        class_index=[47]*4+[19]*2,
        scientific_name=['Hirundo rustica',None,'','Hirundo rustica',None,'Other bird']))


def test_missing_names_do_not_duplicate_taxon_lookup():
    df=fixture_frame()
    old=df.drop_duplicates().set_index('taxon_id')
    with pytest.raises(TypeError): int(old.loc[11901,'class_index'])
    new=canonical_taxon_map(df)
    assert new.index.is_unique and len(new)==2
    assert int(new.loc[11901,'class_index'])==47
    assert new.loc[11901,'scientific_name']=='Hirundo rustica'


@pytest.mark.parametrize('kind',['class_conflict','name_conflict','all_names_missing','noninteger_class','class_alias'])
def test_real_conflicts_still_fail_closed(kind):
    df=fixture_frame()
    if kind=='class_conflict': df.loc[0,'class_index']=48
    if kind=='name_conflict': df.loc[1,'scientific_name']='Different species'
    if kind=='all_names_missing': df.loc[df.taxon_id==11901,'scientific_name']=None
    if kind=='noninteger_class': df['class_index']=df.class_index.astype(float)+.5
    if kind=='class_alias': df['class_index']=47
    with pytest.raises(GateStop): canonical_taxon_map(df)


def test_input_not_modified_and_permutation_invariant():
    df=fixture_frame(); saved=df.copy(deep=True)
    a=canonical_taxon_map(df)
    b=canonical_taxon_map(df.sample(frac=1,random_state=9))
    pd.testing.assert_frame_equal(a,b)
    pd.testing.assert_frame_equal(df,saved)
