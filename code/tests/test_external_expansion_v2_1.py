from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from robird.external_expansion_v2_1 import canonical_taxa
from robird.external_cohort_v1_1 import GateStop


def rows():
    return pd.DataFrame([dict(taxon_id=i+1,class_index=i,scientific_name=f'Taxon {i}') for i in range(100)])


def test_empty_display_names_do_not_create_extra_taxa():
    frame=rows();extra=frame.iloc[:78].copy();extra['scientific_name']=np.nan
    joined=pd.concat([frame,extra],ignore_index=True)
    assert len(joined)==178
    pd.testing.assert_frame_equal(canonical_taxa(joined),frame)


def test_conflicting_names_rejected_not_arbitrarily_selected():
    frame=rows();extra=frame.iloc[:1].copy();extra['scientific_name']='Different species'
    with pytest.raises(GateStop,match='ambiguous'):
        canonical_taxa(pd.concat([frame,extra]))


def test_conflicting_class_identity_rejected():
    frame=rows();frame.loc[0,'taxon_id']=2
    with pytest.raises(GateStop,match='identity'):
        canonical_taxa(frame)


def test_real_frozen_development_mapping_has_exactly_100_classes():
    root=Path(__file__).resolve().parents[2]
    source=pd.read_csv(root/'code/data/manifests/development_photos_v5_3.csv')
    table=canonical_taxa(source)
    pd.testing.assert_frame_equal(table[['taxon_id','class_index']],source[['taxon_id','class_index']].drop_duplicates().sort_values('class_index').reset_index(drop=True))
    assert len(table)==100 and table.scientific_name.notna().all()
