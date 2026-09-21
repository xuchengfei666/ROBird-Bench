"""Only normalize display names; never change species IDs or class indices."""
import pandas as pd
from robird.external_cohort_v1_1 import GateStop


def canonical_taxa(frame):
    mapping=frame[['taxon_id','class_index']].drop_duplicates().sort_values('class_index')
    if len(mapping)!=100 or mapping.taxon_id.nunique()!=100 or mapping.class_index.tolist()!=list(range(100)):
        raise GateStop('Original taxon/class identity mapping is not the fixed 100 classes')
    rows=[]
    for taxon,index in mapping.itertuples(index=False,name=None):
        values=frame.loc[(frame.taxon_id==taxon)&(frame.class_index==index),'scientific_name']
        names=sorted({str(v).strip() for v in values.dropna() if str(v).strip()})
        if len(names)!=1:
            raise GateStop(f'Nonempty display names are absent or ambiguous: taxon {taxon}: {names}')
        rows.append(dict(taxon_id=int(taxon),class_index=int(index),scientific_name=names[0]))
    return pd.DataFrame(rows)
