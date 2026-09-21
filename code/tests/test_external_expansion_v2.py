import copy
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from robird import external_expansion_v2 as m
from robird.io import atomic_write_csv, atomic_write_json, sha256_file
from robird.rsos_suite_v1 import finish


def config():
    return dict(created_d2='2026-09-08T00:00:00Z',per_page=200,min_observers=8,
                sensitivity_min_observers=10,target_observers=20,max_groups_per_observer_taxon=2,
                max_pages_per_taxon=100,request_attempts=5,daily_request_limit=5000,
                request_interval_seconds=2,download_minimum_taxa_exclusive=13)


def observation(oid=100,uid=200,taxon=1,n=2):
    return dict(id=oid,user={'id':uid},taxon={'id':taxon},quality_grade='research',
        created_at='2026-09-07T10:00:00Z',observed_on='2026-09-07',
        photos=[dict(id=oid*10+i,license_code='cc-by',attribution='Test source attribution',
                     url=f'https://inaturalist-open-data.s3.amazonaws.com/photos/{oid*10+i}/square.jpg') for i in range(n)])


def excluded():
    return {key:set() for key in ('photo_id','observation_id','observer_id')}


def test_filter_respects_every_source_identity():
    obs=observation()
    for key,value in [('photo_id',1000),('observation_id',100),('observer_id',200)]:
        ids=excluded();ids[key].add(value)
        group,reason=m.candidate(obs,1,ids,config()['created_d2'])
        assert group is None
    assert m.candidate(obs,1,excluded(),config()['created_d2'])[0] is not None


@pytest.mark.parametrize('change,reason',[
    ({'quality_grade':'casual'},'not_research_grade'),
    ({'taxon':{'id':2}},'not_exact_species_id'),
    ({'created_at':'2026-09-09T00:00:00Z'},'after_creation_cutoff'),
    ({'created_at':'2026-09-01T00:00:00'},'invalid_created_at'),
    ({'created_at':'broken'},'invalid_created_at')])
def test_filter_semantics(change,reason):
    obs=observation();obs.update(change)
    assert m.candidate(obs,1,excluded(),config()['created_d2'])==(None,reason)


def test_per_photo_license_url_and_deduplication():
    obs=observation(n=3);obs['photos'][0]['license_code']='cc-by-nc'
    group,_=m.candidate(obs,1,excluded(),config()['created_d2'])
    assert [p['id'] for p in group['photos']]==[1001,1002]
    obs['photos'][1]['url']='https://evil.invalid/photos/123/square.jpg'
    assert m.candidate(obs,1,excluded(),config()['created_d2'])[0] is None
    obs=observation(n=2);obs['photos'][1]=dict(obs['photos'][0])
    assert m.candidate(obs,1,excluded(),config()['created_d2'])[0] is None


def test_sorted_first_five_and_extensionless_url():
    obs=observation(n=7);obs['photos'].reverse()
    obs['photos'][0]['url']=obs['photos'][0]['url'].replace('square.jpg','square.')
    group,_=m.candidate(obs,1,excluded(),config()['created_d2'])
    assert [p['id'] for p in group['photos']]==list(range(1000,1005))
    assert m.legacy.large_url(obs['photos'][0]['url']).endswith('large.')


def test_cursor_and_observer_caps():
    cfg=config();cfg['target_observers']=3
    rows=[observation(100,1),observation(99,1),observation(98,1),observation(97,2),observation(96,3),observation(95,4)]
    state=m.fresh_state();cursor,exhausted=m.consume_page({'results':rows},1,cfg,excluded(),state,101)
    assert cursor==95 and exhausted
    assert len(state['candidates'])==4 and len(state['observers'])==3
    assert state['attrition']['observer_taxon_group_cap']==1
    assert state['processed_records']==5 and state['returned_records']==6
    with pytest.raises(m.GateStop,match='cursor'):
        m.consume_page({'results':rows},1,cfg,excluded(),m.fresh_state(),100)


def test_cursor_never_uses_numbered_deep_pages():
    params=m.params_for(config(),99,cursor=123456)
    assert params['id_below']==123456 and 'page' not in params
    assert params['created_d2']==config()['created_d2']


def test_unordered_api_response_rejected():
    with pytest.raises(m.GateStop,match='descending'):
        m.consume_page({'results':[observation(1),observation(2)]},1,config(),excluded(),m.fresh_state())


def test_legacy_overlap_not_counted_twice():
    state=m.fresh_state();page={'results':[observation(100,1),observation(99,2)]}
    m.consume_page(page,1,config(),excluded(),state)
    m.consume_page(page,1,config(),excluded(),state)
    assert len(state['candidates'])==2 and state['attrition']['duplicate_pagination_id']==2


def test_cached_page_is_resumed_and_hash_checked(tmp_path):
    params=m.params_for(config(),1,cursor=101)
    payload=json.dumps({'results':[observation()]}).encode()
    path=tmp_path/'pages/1/006.json.gz'
    m.atomic_binary(path,lambda f:f.write(gzip.compress(payload,mtime=0)))
    finish(path.with_suffix('.done.json'),'contract',[path],params=params,retrieved_at='test',source='test')
    value,_,_=m.get_page(tmp_path,1,6,params,'contract',config(),None)
    assert value['results'][0]['id']==100
    with pytest.raises(m.GateStop,match='parameters'):
        m.get_page(tmp_path,1,6,dict(params,id_below=102),'contract',config(),None)
    m.atomic_binary(path,lambda f:f.write(b'changed'))
    with pytest.raises(RuntimeError,match='changed'):
        m.get_page(tmp_path,1,6,params,'contract',config(),None)


def fixture_census(tmp_path,n=10):
    cfg=config();cfg.update(taxon_manifest='taxa.csv',exclusion_manifests=['history.csv'],
                            cached_census=str(tmp_path/'cache'))
    atomic_write_csv(tmp_path/'taxa.csv',pd.DataFrame([dict(taxon_id=i+1,class_index=i,scientific_name=f'Taxon {i}') for i in range(100)]))
    atomic_write_csv(tmp_path/'history.csv',pd.DataFrame(columns=['photo_id','observation_id','observer_id']))
    cache=tmp_path/'cache';cache.mkdir()
    for tid in range(1,101):
        page=cache/f'response_{tid}_1.json'
        rows=[observation(oid=tid*10000+1000-j,uid=tid*1000+j,taxon=tid) for j in range(n)]
        atomic_write_json(page,dict(results=rows,total_results=n))
        finish(page.with_suffix('.done.json'),'legacy',[page],params=m.params_for(cfg,tid,page=1))
    return cfg


def test_whole_census_selection_idempotence_and_contract(tmp_path):
    cfg=fixture_census(tmp_path,10);data=tmp_path/'data';report=tmp_path/'report'
    summary=m.census(tmp_path,data,report,cfg,'test-contract',lambda **kw:None)
    assert summary['taxa']==100 and summary['eligible8']==100 and summary['eligible10']==100
    assert summary['stop_reasons']=={'SOURCE_EXHAUSTED':100}
    frame=m.select_metadata(tmp_path,report,cfg,'test-contract',summary)
    assert len(frame)==2000 and frame.observation_id.nunique()==1000
    assert frame.strict10_taxon.all()
    digest=sha256_file(report/'metadata_done.json')
    again=m.census(tmp_path,data,report,cfg,'test-contract',lambda **kw:None)
    pd.testing.assert_frame_equal(frame,m.select_metadata(tmp_path,report,cfg,'test-contract',again))
    assert sha256_file(report/'metadata_done.json')==digest
    with pytest.raises(RuntimeError,match='Contract mismatch'):
        m.census(tmp_path,data,report,cfg,'different-contract',lambda **kw:None)


def test_eight_is_not_ten(tmp_path):
    cfg=fixture_census(tmp_path,8);report=tmp_path/'report'
    summary=m.census(tmp_path,tmp_path/'data',report,cfg,'eight',lambda **kw:None)
    assert summary['eligible8']==100 and summary['eligible10']==0
    frame=m.select_metadata(tmp_path,report,cfg,'eight',summary)
    assert not frame.strict10_taxon.any()


def test_no_download_authority_when_not_expanded(tmp_path):
    with pytest.raises(m.GateStop,match='NOT_EXPANDED'):
        m.select_metadata(tmp_path,tmp_path/'report',config(),'test',dict(eligible8=13,reports=[]))
    assert not (tmp_path/'report/metadata_done.json').exists()


def test_models_cannot_run_before_duplicate_and_byte_gate(tmp_path):
    with pytest.raises(m.GateStop,match='incomplete data gate'):
        m.require_data_gates(pd.DataFrame(),tmp_path,tmp_path,tmp_path,config(),'test')


def test_missing_high_budgets_are_not_zero_or_dropped():
    rows=m.paired_records({},dict(mode='strict10',model='deepsets',training_policy='all'),99)
    assert len(rows)==4 and all(row['observer_signflip_p']==1 for row in rows)
    assert all('NOT_ESTIMABLE' in row['status'] for row in rows)


def test_quota_committed_before_request(tmp_path,monkeypatch):
    cfg=config();cfg['request_interval_seconds']=0
    budget=m.RequestBudget(tmp_path,cfg,lambda **kw:None)
    budget.take();budget.take()
    assert sum(m.read_json(tmp_path/'api_quota.json')['days'].values())==2


def test_freeze_drift_rejected(tmp_path):
    path=tmp_path/'input.json';atomic_write_json(path,{'a':1})
    files={str(path):sha256_file(path)};m.verify_files(files)
    atomic_write_json(path,{'a':2})
    with pytest.raises(m.GateStop,match='Frozen input changed'):
        m.verify_files(files)


def deep_prefix(tmp_path):
    cfg=fixture_census(tmp_path,10)
    atomic_write_csv(tmp_path/'history.csv',pd.DataFrame([dict(photo_id=-1,observation_id=-1,observer_id=999)]))
    for number in range(1,6):
        rows=[observation(oid=11000-(number-1)*200-j,uid=999,taxon=1) for j in range(200)]
        path=tmp_path/f'cache/response_1_{number}.json'
        atomic_write_json(path,dict(results=rows,total_results=5000))
        # Replacing synthetic fixture markers only; production markers are immutable.
        atomic_write_json(path.with_suffix('.done.json'),dict(contract='legacy',
            artifacts={str(path.resolve()):sha256_file(path)},params=m.params_for(cfg,1,page=number)))
    return cfg


def test_real_census_continues_after_legacy_five_pages(tmp_path,monkeypatch):
    cfg=deep_prefix(tmp_path);calls=[]
    def fake_page(directory,taxon,number,params,contract,config,budget):
        calls.append((taxon,number,params))
        assert taxon==1 and number==6 and params['id_below']==10001 and 'page' not in params
        value=dict(results=[observation(oid=10000-i,uid=30000+i,taxon=1) for i in range(10)])
        path=directory/'test_page6.json';atomic_write_json(path,value)
        return value,[path],dict(source='synthetic',params=params,retrieved_at='test')
    monkeypatch.setattr(m,'get_page',fake_page)
    summary=m.census(tmp_path,tmp_path/'data',tmp_path/'report',cfg,'deep-test',lambda **kw:None)
    row=summary['reports'][0]
    assert len(calls)==1 and row['pages']==6 and row['independent_observers']==10
    assert row['first_failure_attrition']['historical_observer']==1000
    assert row['stop_reason']=='SOURCE_EXHAUSTED'


def test_budget_truncation_is_not_source_exhaustion(tmp_path):
    cfg=deep_prefix(tmp_path);cfg['max_pages_per_taxon']=5
    summary=m.census(tmp_path,tmp_path/'data',tmp_path/'report',cfg,'truncated-test',lambda **kw:None)
    row=summary['reports'][0]
    assert row['stop_reason']=='SEARCH_BUDGET_TRUNCATED' and row['independent_observers']==0
    assert summary['eligible8']==99
