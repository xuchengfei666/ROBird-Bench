"""Independent, prediction-blind external expansion; frozen parents stay read-only."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from robird.io import atomic_write_csv, atomic_write_json, sha256_file
from robird.rsos_suite_v1 import atomic_binary, finish, marker
from robird.artifact_tree_v1 import verify_tree
from robird import external_cohort_v1_1 as legacy

GateStop = legacy.GateStop
read_json = legacy.read_json
LICENSES = ('cc0', 'cc-by', 'cc-by-sa')
API = 'https://api.inaturalist.org/v1/observations'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def params_for(config, taxon, cursor=None, page=None):
    params = dict(taxon_id=int(taxon), quality_grade='research', photos='true',
                  photo_license=','.join(LICENSES), created_d2=config['created_d2'],
                  order_by='id', order='desc', per_page=config['per_page'])
    if cursor is not None:
        params['id_below'] = int(cursor)
    if page is not None:
        params['page'] = int(page)
    return params


def verify_files(files):
    for name, expected in files.items():
        if sha256_file(Path(name)) != expected:
            raise GateStop('Frozen input changed: '+name)


def history_ids(root, config):
    history = [pd.read_csv(root/p, keep_default_na=False) for p in config['exclusion_manifests']]
    excluded = {c: set() for c in ('photo_id', 'observation_id', 'observer_id')}
    for frame in history:
        for column, values in excluded.items():
            if column in frame:
                values.update(pd.to_numeric(frame[column].replace('', np.nan)).dropna().astype('int64'))
    return history, excluded


def candidate(observation, taxon, excluded, cutoff):
    """Return first-failure reason; photo IDs remain source IDs, not dataset row IDs."""
    try:
        oid, uid = int(observation['id']), int(observation['user']['id'])
        tid = int((observation.get('taxon') or {}).get('id', -1))
    except (KeyError, TypeError, ValueError):
        return None, 'invalid_source_identity'
    if oid in excluded['observation_id']:
        return None, 'historical_observation'
    if uid in excluded['observer_id']:
        return None, 'historical_observer'
    if tid != taxon:
        return None, 'not_exact_species_id'
    if observation.get('quality_grade') != 'research':
        return None, 'not_research_grade'
    try:
        created = datetime.fromisoformat(observation['created_at'].replace('Z', '+00:00'))
        boundary = datetime.fromisoformat(cutoff.replace('Z', '+00:00'))
        if created.tzinfo is None:
            return None, 'invalid_created_at'
        if created > boundary:
            return None, 'after_creation_cutoff'
    except (KeyError, ValueError, TypeError):
        return None, 'invalid_created_at'
    photos, seen = [], set()
    for photo in observation.get('photos', []):
        try:
            pid = int(photo['id'])
            if pid in seen or pid in excluded['photo_id']:
                continue
            if photo.get('license_code') not in LICENSES or not photo.get('attribution'):
                continue
            legacy.large_url(photo['url'])
        except (KeyError, ValueError, TypeError, GateStop):
            continue
        seen.add(pid)
        photos.append(dict(id=pid, url=photo['url'], license_code=photo['license_code'],
                           attribution=photo['attribution']))
    photos = sorted(photos, key=lambda p: p['id'])[:5]
    if len(photos) < 2:
        return None, 'fewer_than_two_usable_photos'
    return dict(observation_id=oid, observer_id=uid, taxon_id=tid,
                created_at=observation['created_at'], observed_on=observation.get('observed_on'),
                photos=photos), 'eligible_before_observer_cap'


def consume_page(response, taxon, config, excluded, state, cursor=None):
    records = response.get('results')
    if not isinstance(records, list) or len(records) > config['per_page']:
        raise GateStop('Malformed/oversized API page')
    ids = [int(r['id']) for r in records]
    if ids != sorted(ids, reverse=True) or len(ids) != len(set(ids)):
        raise GateStop('Response IDs not strictly descending/unique')
    if cursor is not None and any(i >= cursor for i in ids):
        raise GateStop('id_below cursor was not obeyed')
    state['returned_records'] += len(records)
    for obs in records:
        if len(state['observers']) >= config['target_observers']:
            break
        oid = int(obs['id'])
        state['processed_records'] += 1
        if oid in state['seen']:
            state['attrition']['duplicate_pagination_id'] += 1
            continue
        state['seen'].add(oid)
        group, reason = candidate(obs, taxon, excluded, config['created_d2'])
        if group is None:
            state['attrition'][reason] += 1
            continue
        observer = group['observer_id']
        if state['observers'][observer] >= config['max_groups_per_observer_taxon']:
            state['attrition']['observer_taxon_group_cap'] += 1
            continue
        state['observers'][observer] += 1
        state['candidates'].append(group)
        state['attrition']['retained_groups'] += 1
    return min(ids) if ids else cursor, len(records) < config['per_page']


def fresh_state():
    return dict(seen=set(), observers=Counter(), attrition=Counter(), candidates=[],
                returned_records=0, processed_records=0)


class RequestBudget:
    """Quota is committed BEFORE every attempt, including failed attempts and restarts."""
    def __init__(self, directory, config, progress):
        self.path = directory/'api_quota.json'
        self.config, self.progress = config, progress

    def take(self):
        while True:
            day = datetime.now(timezone.utc).date().isoformat()
            saved = read_json(self.path) if self.path.exists() else dict(days={}, last_request=0)
            count = saved['days'].get(day, 0)
            if count < self.config['daily_request_limit']:
                pause = self.config['request_interval_seconds']-(time.time()-saved['last_request'])
                if pause > 0:
                    time.sleep(pause)
                saved['days'][day] = count+1
                saved['last_request'] = time.time()
                atomic_write_json(self.path, saved)
                return
            self.progress(status='WAITING_API_DAILY_QUOTA', day=day, requests=count)
            time.sleep(60)


def get_page(directory, taxon, number, params, contract, config, budget):
    path = directory/'pages'/str(taxon)/f'{number:03d}.json.gz'
    done = path.with_suffix('.done.json')
    saved = marker(done, contract)
    if saved:
        if saved['params'] != params:
            raise GateStop('Committed pagination parameters changed')
        return json.loads(gzip.decompress(path.read_bytes())), [path, done], saved
    if shutil.disk_usage(directory).free < 10*1024**3:
        raise GateStop('Less than 10GiB free on data drive')
    url = API+'?'+urlencode(params)
    for attempt in range(config['request_attempts']):
        budget.take()
        try:
            request = Request(url, headers={'User-Agent': 'ROBird-external-expansion/2 academic-research'})
            with urlopen(request, timeout=90) as response:
                payload = response.read(128*1024*1024+1)
            if len(payload) > 128*1024*1024:
                raise GateStop('API response exceeds 128MiB bound')
            value = json.loads(payload)
            if not isinstance(value.get('results'), list):
                raise GateStop('API response lacks results')
            break
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            transient = not isinstance(exc, HTTPError) or exc.code == 429 or exc.code >= 500
            if not transient or attempt+1 == config['request_attempts']:
                raise
            delay = min(300, 10*2**attempt)
            retry_after = exc.headers.get('Retry-After') if isinstance(exc, HTTPError) else None
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    from email.utils import parsedate_to_datetime
                    delay = max(delay, parsedate_to_datetime(retry_after).timestamp()-time.time())
            time.sleep(max(0, delay))
    atomic_binary(path, lambda f: f.write(gzip.compress(payload, compresslevel=6, mtime=0)))
    saved = finish(done, contract, [path], params=params, request_url=url, retrieved_at=utc_now(),
                   response_sha256=hashlib.sha256(payload).hexdigest(), source='live_cursor_continuation')
    return value, [path, done], saved


def census(root, data, report, config, contract, progress):
    done = report/'census_done.json'
    if marker(done, contract):
        verify_tree(done)
        return read_json(report/'census_summary.json')
    directory = data/'metadata'
    directory.mkdir(parents=True, exist_ok=True)
    budget = RequestBudget(directory, config, progress)
    _, excluded = history_ids(root, config)
    names = pd.read_csv(root/config['taxon_manifest'])[['taxon_id','class_index','scientific_name']].drop_duplicates().sort_values('class_index')
    if len(names) != 100 or names.taxon_id.nunique() != 100 or names.class_index.tolist() != list(range(100)):
        raise GateStop('Expected original contiguous 100-taxon mapping')
    reports, artifacts = [], []
    cache = Path(config['cached_census'])
    for item in names.itertuples(index=False):
        taxon = int(item.taxon_id)
        target = directory/f'taxon_{taxon}.json'
        saved = marker(target, contract)
        if saved:
            verify_tree(target)
            reports.append(saved['result']); artifacts.append(target)
            continue
        state = fresh_state()
        paths, sources, cursor, ending = [], [], None, None
        for number in range(1, config['max_pages_per_taxon']+1):
            old = cache/f'response_{taxon}_{number}.json'
            if number <= 5 and old.exists():
                commit = old.with_suffix('.done.json')
                old_marker = read_json(commit)
                expected = params_for(config, taxon, page=number)
                if old_marker['params'] != expected:
                    raise GateStop('Cached query does not match new fixed query')
                verify_tree(commit)
                response = read_json(old)
                page_paths = [old, commit]
                source = dict(source='hash_verified_legacy_cache', params=expected,
                              path=str(old), sha256=sha256_file(old),
                              retrieved_at='legacy_2026-09-08_census_exact_time_unrecorded')
                checked_cursor = None  # Legacy numbered pages may overlap; IDs are deduplicated.
            else:
                if cursor is None:
                    raise GateStop('Missing legacy first-page cursor')
                params = params_for(config, taxon, cursor=cursor)
                response, page_paths, source = get_page(directory, taxon, number, params, contract, config, budget)
                checked_cursor = cursor
            cursor, exhausted = consume_page(response, taxon, config, excluded, state, checked_cursor)
            paths.extend(page_paths)
            sources.append(dict(page_number=number, source=source['source'], params=source['params'],
                                retrieved_at=source['retrieved_at'], returned=len(response['results']),
                                query_total_results=response.get('total_results'), cursor_after=cursor))
            progress(completed_taxa=len(reports), total_taxa=100, taxon_id=taxon, page=number,
                     observers=len(state['observers']), retained_groups=len(state['candidates']))
            if len(state['observers']) >= config['target_observers']:
                ending = 'TARGET_REACHED'; break
            if exhausted:
                ending = 'SOURCE_EXHAUSTED'; break
        if ending is None:
            ending = 'SEARCH_BUDGET_TRUNCATED'
        n = len(state['observers'])
        result = dict(taxon_id=taxon, class_index=int(item.class_index), scientific_name=item.scientific_name,
                      stop_reason=ending, independent_observers=n, candidate_groups=len(state['candidates']),
                      eligible8=n >= config['min_observers'], eligible10=n >= config['sensitivity_min_observers'],
                      pages=len(sources), returned_records=state['returned_records'], processed_records=state['processed_records'],
                      first_failure_attrition=dict(state['attrition']), sources=sources, candidates=state['candidates'])
        finish(target, contract, paths, result=result)
        reports.append(result); artifacts.append(target)
    summary = dict(status='EXPANSION_METADATA_CENSUS_COMPLETE', taxa=100,
                   eligible8=sum(r['eligible8'] for r in reports), eligible10=sum(r['eligible10'] for r in reports),
                   stop_reasons=dict(Counter(r['stop_reason'] for r in reports)),
                   exclusion_counts={k:len(v) for k,v in excluded.items()},
                   original_g6_pass=False, predictions_seen_for_selection=False,
                   mixed_metadata_retrieval_dates=True, reports=reports)
    atomic_write_json(report/'census_summary.json', summary)
    finish(done, contract, artifacts+[report/'census_summary.json'], status=summary['status'])
    return summary


def select_metadata(root, report, config, contract, summary):
    path, done = report/'metadata.csv', report/'metadata_done.json'
    if marker(done, contract):
        verify_tree(done)
        return pd.read_csv(path, keep_default_na=False)
    coverage = [{k:v for k,v in r.items() if k not in ('sources','candidates','first_failure_attrition')}
                | {'attrition_json':json.dumps(r['first_failure_attrition'], sort_keys=True)} for r in summary['reports']]
    atomic_write_csv(report/'feasibility_100_taxa.csv', pd.DataFrame(coverage))
    if summary['eligible8'] <= config['download_minimum_taxa_exclusive']:
        raise GateStop(f"NOT_EXPANDED: only {summary['eligible8']} eligible taxa; no download authorized")
    rows = []
    for taxon in summary['reports']:
        if not taxon['eligible8']:
            continue
        for group in taxon['candidates']:
            for photo in group['photos']:
                rows.append(dict(taxon_id=taxon['taxon_id'], class_index=taxon['class_index'],
                    scientific_name=taxon['scientific_name'], observation_id=group['observation_id'],
                    observer_id=group['observer_id'], photo_id=photo['id'], license_code=photo['license_code'],
                    attribution=photo['attribution'], observed_on=group['observed_on'], created_at=group['created_at'],
                    original_url=photo['url'], url=legacy.large_url(photo['url']),
                    observation_url=f"https://www.inaturalist.org/observations/{group['observation_id']}",
                    cohort='INDEPENDENT_EXPANDED_SAME_PLATFORM_COHORT_V2',
                    split='external_feasibility_test', selection_basis='frozen_metadata_only_v2',
                    strict10_taxon=bool(taxon['eligible10'])))
    frame = pd.DataFrame(rows).sort_values(['taxon_id','observation_id','photo_id']).reset_index(drop=True)
    history, _ = history_ids(root, config)
    legacy.assert_metadata(frame, history)
    support = frame.groupby('taxon_id').observer_id.nunique()
    if len(support) != summary['eligible8'] or (support < config['min_observers']).any():
        raise GateStop('Selected cohort observer support changed')
    atomic_write_csv(path, frame)
    atomic_write_csv(report/'cardinality_coverage.csv', frame.groupby(['taxon_id','observation_id']).size()
                     .rename('photos').reset_index().groupby(['taxon_id','photos']).size().rename('groups').reset_index())
    finish(done, contract, [path,report/'census_done.json',report/'feasibility_100_taxa.csv',report/'cardinality_coverage.csv'],
           taxa=len(support), groups=frame.observation_id.nunique(), photos=len(frame),
           strict10_taxa=summary['eligible10'], original_g6_pass=False,
           predictions_seen_for_selection=False, frozen_before_download=True)
    return frame


def require_data_gates(frame, root, data, report, config, contract):
    for name in ('metadata_done.json','download_done.json','duplicate_done.json'):
        if not marker(report/name, contract):
            raise GateStop('Model blocked by incomplete data gate: '+name)
        verify_tree(report/name)
    history, _ = history_ids(root, config)
    legacy.assert_metadata(frame, history)
    if not frame.license_code.isin(LICENSES).all() or not frame.attribution.map(bool).all():
        raise GateStop('License/provenance gate changed')
    for row in frame.itertuples():
        ledger = marker(data/'download_ledger'/f'{row.photo_id}.json', contract)
        if not ledger or ledger['details']['sha256'] != row.sha256:
            raise GateStop('Missing/changed download ledger')
        if sha256_file(Path(row.local_path)) != row.sha256:
            raise GateStop('Image bytes changed after duplicate screen')


def paired_records(combined, identity, repeats):
    from robird.external_completion_v1 import observer_signflip, TRANSITIONS
    from robird.budget_metrics_v2_1 import cluster_interval
    results = []
    for before, after in TRANSITIONS:
        base = dict(identity, before=before, after=after)
        if before not in combined or after not in combined:
            results.append(dict(base,status='NOT_ESTIMABLE_NO_ELIGIBLE_OBSERVATIONS',
                                observer_signflip_p=1.,taxon_signflip_p=1.)); continue
        ma, va = combined[before]; mb, vb = combined[after]
        a = ma[['observation_id','observer_id','label','taxon_id']].copy()
        a['before_accuracy'] = va[:,:,0].mean(0)
        b = mb[['observation_id']].copy(); b['after_accuracy'] = vb[:,:,0].mean(0)
        pair = a.merge(b,on='observation_id',validate='one_to_one')
        pair['delta'] = pair.after_accuracy-pair.before_accuracy
        count = pair.groupby('label').delta.transform('size'); n = pair.label.nunique()
        weighted = pair.delta/(n*count)
        effects = weighted.groupby(pair.observer_id).sum()
        test = observer_signflip(effects.to_numpy(), repeats)
        results.append(dict(base,status='ESTIMATED',groups=len(pair),taxa=n,observers=len(effects),
            delta_accuracy=test['statistic'],observer_signflip_p=test['p'],test_method=test['method'],
            taxon_signflip_p=legacy.signflip(pair,'delta'),observer_interval=cluster_interval(pair,'delta',5000),
            taxon_interval=legacy.taxon_interval(pair,'delta'),
            independent_source_observers_and_sign_exchangeability='ASSUMED_NOT_PROVEN',
            predictions_used_to_choose_cohort=False))
    return results


def statistics(frame, data, report, config, contract):
    from robird.external_completion_v1 import components, combine_seeds, interval_table
    done = report/'statistics_done.json'
    if marker(done, contract):
        verify_tree(done); return
    tables, tests, artifacts = [], [], []
    for mode in ('full8','strict10'):
        manifest = frame if mode == 'full8' else frame[frame.strict10_taxon.astype(str).str.lower().eq('true')]
        for model in legacy.MODELS:
            for policy in ('all','k1'):
                specs = sorted((s for s in config['runs'] if s['model']==model and s['training_policy']==policy),
                               key=lambda s:s['seed'])
                directory = report/'statistics'/mode/f'{model}-{policy}'
                mark = directory/'done.json'
                if not marker(mark, contract):
                    identity = dict(mode=mode, model=model, training_policy=policy)
                    parts, packets = [], []
                    if len(manifest):
                        for spec in specs:
                            raw = data/'runs'/spec['run_id']/'predictions.csv'
                            predictions = pd.read_csv(raw)
                            predictions = predictions[predictions.observation_id.isin(manifest.observation_id)]
                            packet = components(predictions, manifest)
                            packets.append(packet)
                            parts.append(interval_table(packet,dict(identity,seed=spec['seed']),config['bootstrap_repeats']))
                        combined = combine_seeds(packets)
                        parts.append(interval_table(combined,dict(identity,seed='mean_of_3'),config['bootstrap_repeats']))
                    else:
                        combined = {}
                    table = pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['status','mode','model'])
                    atomic_write_csv(directory/'intervals.csv',table)
                    atomic_write_json(directory/'paired.json',paired_records(combined,identity,config['signflip_repeats']))
                    finish(mark,contract,[directory/'intervals.csv',directory/'paired.json'],taxa=manifest.taxon_id.nunique())
                tables.append(pd.read_csv(directory/'intervals.csv'))
                tests.extend(read_json(directory/'paired.json')); artifacts.append(mark)
                atomic_write_json(report/'statistics_progress.json',dict(mode=mode,model=model,policy=policy))
    if len(tests)!=64:
        raise GateStop('Fixed 64-comparison family incomplete')
    for field in ('observer_signflip_p','taxon_signflip_p'):
        for row,p in zip(tests,legacy.holm([r[field] for r in tests])):
            row[field+'_holm64']=float(p)
    atomic_write_csv(report/'all_metric_intervals.csv',pd.concat(tables,ignore_index=True))
    atomic_write_json(report/'paired_budget_tests.json',tests)
    finish(done,contract,artifacts+[report/'all_metric_intervals.csv',report/'paired_budget_tests.json'],
           original_g6_pass=False, human_quality_ratings='NOT_PERFORMED_EXCLUDED_BY_USER_SCOPE_CHANGE',
           seed_estimand='mean_of_seed_metrics_not_ensemble', planned_family_size=64,
           no_estimate_p_placeholder=1., intervals='pointwise_fixed_available_taxa_not_population_unbiased')
