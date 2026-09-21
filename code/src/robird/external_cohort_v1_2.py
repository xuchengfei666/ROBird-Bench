"""Isolated metadata identity correction; all other v1.1 behavior is reused."""
from pathlib import Path
import pandas as pd
from robird.external_cohort_v1_1 import (
    GateStop, SCOPE, marker, verify_tree, read_json, large_url, assert_metadata,
    atomic_write_csv, finish,
)


def canonical_taxon_map(dev):
    required = ['taxon_id', 'class_index', 'scientific_name']
    if not set(required).issubset(dev.columns):
        raise GateStop('Missing taxonomy fields')
    rows = []
    for taxon, part in dev.groupby('taxon_id', dropna=False):
        if pd.isna(taxon) or int(taxon) != taxon:
            raise GateStop('Invalid taxon ID')
        labels = pd.to_numeric(part.class_index, errors='raise')
        if labels.isna().any() or labels.nunique() != 1 or (labels % 1 != 0).any():
            raise GateStop(f'Conflicting/invalid class indices for taxon {taxon}')
        names = part.scientific_name.dropna().astype(str).str.strip()
        names = sorted(set(names[names != '']))
        if len(names) != 1:
            raise GateStop(f'Missing or conflicting nonempty names for taxon {taxon}: {names}')
        rows.append(dict(taxon_id=int(taxon), class_index=int(labels.iloc[0]),
                         scientific_name=names[0]))
    result = pd.DataFrame(rows).set_index('taxon_id').sort_index()
    if result.class_index.duplicated().any():
        raise GateStop('One class index maps to multiple taxa')
    return result


def materialize(root, data, report, config, contract):
    done = report/'metadata_done.json'
    path = report/'metadata.csv'
    if marker(done, contract):
        return pd.read_csv(path, keep_default_na=False)
    meta = Path(config['census'])
    verify_tree(meta/'done.json')
    census = read_json(meta/'done.json')
    if census['taxa'] != 100 or census['individually_feasible'] != 13:
        raise GateStop('Census coverage changed')
    old = pd.read_csv(root/'code/data/manifests/external_cohort_v1.csv')
    dev = pd.read_csv(root/'code/data/manifests/development_photos_v5_3.csv')
    names = canonical_taxon_map(dev)
    identities = set(zip(old.taxon_id, old.observation_id, old.photo_id))
    records, feasibility = [], []
    for p in sorted(meta.glob('taxon_*.json')):
        result = read_json(p)['result']
        taxon = int(result['taxon_id'])
        feasible = result['independent_observers'] >= 10
        feasibility.append(dict(taxon_id=taxon, scientific_name=names.loc[taxon, 'scientific_name'],
                                independent_observers=result['independent_observers'],
                                candidate_groups=result['candidate_groups'],
                                selected_by_metadata=feasible, shortfall=result['shortfall']))
        if not feasible:
            continue
        for group in result['candidates']:
            for photo in group['photos']:
                if (taxon, group['observation_id'], photo['id']) not in identities:
                    raise GateStop('Source materialization membership mismatch')
                if photo.get('license_code') not in ('cc0', 'cc-by', 'cc-by-sa') or not photo.get('attribution'):
                    raise GateStop('Invalid license/attribution')
                records.append(dict(taxon_id=taxon, class_index=int(names.loc[taxon, 'class_index']),
                    scientific_name=names.loc[taxon, 'scientific_name'],
                    observation_id=group['observation_id'], observer_id=group['observer_id'],
                    photo_id=photo['id'], license_code=photo['license_code'], attribution=photo['attribution'],
                    observed_on=group.get('observed_on'), created_at=group.get('created_at'),
                    original_url=photo['url'], url=large_url(photo['url']),
                    observation_url=f"https://www.inaturalist.org/observations/{group['observation_id']}",
                    cohort=SCOPE, selection_basis='metadata_feasibility_after_census',
                    split='external_feasibility_test'))
    frame = pd.DataFrame(records).sort_values(['taxon_id','observation_id','photo_id']).reset_index(drop=True)
    if len(frame) != 970 or frame.observation_id.nunique() != 339 or frame.taxon_id.nunique() != 13:
        raise GateStop('Expected all 339 groups/970 photos/13 taxa')
    history = [pd.read_csv(root/p) for p in config['exclusion_manifests']]
    assert_metadata(frame, history)
    atomic_write_csv(path, frame)
    atomic_write_csv(report/'feasibility_100_taxa.csv', pd.DataFrame(feasibility))
    atomic_write_csv(report/'cardinality_coverage.csv', frame.groupby(['taxon_id','observation_id']).size()
                     .rename('photos').reset_index().groupby(['taxon_id','photos']).size().rename('groups').reset_index())
    finish(done, contract, [path, report/'feasibility_100_taxa.csv', report/'cardinality_coverage.csv'],
           scope=SCOPE, original_g6_pass=False, photos=970, groups=339, taxa=13)
    return frame


