"""Collect compressed lookup and verify complete whitelist/schema, without labels or models."""
from pathlib import Path
import sys,json,hashlib,base64,gzip,math,shlex
R=Path(__file__).parent;plan=json.loads((R/'plan.json').read_text());sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();mount=plan['runtime']['mount_id']
def call(n,a):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':mount,**a}});assert not r.get('isError'),r;return r['structuredContent']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();job=json.loads((R/'job.json').read_text());receipt={s:call('read_process_output',{'process_id':job['result']['id'],'stream':s,'cursor':0,'limit':32768}) for s in ['stdout','stderr']};assert all(not v['running'] for v in receipt.values()),'CPU job not yet terminal';(R/'managed-process-terminal.json').write_text(json.dumps(receipt,indent=2)+'\n')
code="from pathlib import Path;import json,hashlib;p=Path('output/summary.json');s=json.loads(p.read_text());assert s['status']=='PASS_label_free_all_oldpool_span_top3_lookup';print(json.dumps({'summary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'summary':s}))"
r=call('run_command',{'command':plan['runtime']['python']+' -c '+shlex.quote(code),'cwd':plan['runtime']['cwd'],'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':30});assert r['exit_code']==0,r;meta=json.loads(r['stdout']);summary=meta['summary'];assert summary['plan_sha256']==sha(R/'plan.json') and summary['source_sha256']==sha(R/'extract_lookup.py') and summary['seconds']<120
OUT=R/'output';OUT.mkdir(exist_ok=True);sdata=(json.dumps(summary,ensure_ascii=False,indent=2)+'\n').encode();assert hashlib.sha256(sdata).hexdigest()==meta['summary_sha256'];(OUT/'summary.json').write_bytes(sdata)
remote='public-phenotagger-top3-lookup/output/lookup.jsonl.gz';expected=summary['output'];total=expected['bytes'];actual_bytes=0
with (OUT/'lookup.jsonl.gz').open('wb') as f:
 for offset in range(0,total,262144):
  b=call('read_file',{'path':remote,'offset':offset,'limit':min(262144,total-offset)});assert b['encoding']=='base64';data=base64.b64decode(b['content']);assert len(data)==min(262144,total-offset);f.write(data);actual_bytes+=len(data)
assert actual_bytes==total and sha(OUT/'lookup.jsonl.gz')==expected['sha256']
whitelist=[json.loads(x) for x in (R/'span-whitelist.jsonl').read_text().splitlines()];keys={(x['pmc_id'],x['offset'],x['length']) for x in whitelist};labels=(R.parent/'public-phenotagger-reuse/sources/03-lable.vocab').read_text().strip().splitlines();mapping=json.loads((R.parent/'public-phenotagger-reuse/original-column-to-current-HPO.json').read_text());bycol={x['column']:x['current_label'] for x in mapping['mapping']};seen=set();hasher=hashlib.sha256();ubytes=0;model_entries={'pubmed':0,'bioformer':0}
with gzip.open(OUT/'lookup.jsonl.gz','rb') as f:
 for line in f:
  hasher.update(line);ubytes+=len(line);row=json.loads(line);key=(row['pmc_id'],row['offset'],row['length']);assert key in keys and key not in seen;seen.add(key)
  for model in ['pubmed','bioformer']:
   assert type(row[model+'_raw_present']) is bool;assert row[model+'_raw_occurrences']>=0;assert row[model+'_raw_present']==(row[model+'_raw_occurrences']>0);assert row[model+'_raw_present'] or not row[model];assert len({x['hpo_id'] for x in row[model]})==len(row[model])
   for x in row[model]:
    assert x['observed'] is True and x['observations'];assert x['p']==max(o['p'] for o in x['observations']) and x['rank']==min(o['rank'] for o in x['observations']);assert 1<=x['rank']<=3 and math.isfinite(x['p']) and 0<=x['p']<=1
    for o in x['observations']:assert labels[o['column']]==o['raw_hpo_id'] and bycol[o['column']]==x['hpo_id'] and x['hpo_id'] is not None and 1<=o['rank']<=3 and math.isfinite(o['p']) and 0<=o['p']<=1
    model_entries[model]+=1
assert seen==keys and len(seen)==expected['rows'] and ubytes==expected['jsonl_bytes'] and hasher.hexdigest()==expected['jsonl_sha256'];assert all(model_entries[m]==summary['model_statistics'][m]['current_id_entries'] for m in model_entries)
report={'status':'PASS_complete_whitelist_lookup_transfer_and_semantic_contract','plan_sha256':sha(R/'plan.json'),'extractor_sha256':sha(R/'extract_lookup.py'),'collector_sha256':sha(__file__),'summary_sha256':sha(OUT/'summary.json'),'gzip_sha256':sha(OUT/'lookup.jsonl.gz'),'gzip_bytes':total,'jsonl_sha256':hasher.hexdigest(),'jsonl_bytes':ubytes,'whitelist_sha256':sha(R/'span-whitelist.jsonl'),'spans':len(seen),'model_id_entries':model_entries,'all_whitelist_keys_exactly_once_including_misses':True,'all_aggregates_match_observed_original_columns':True,'no_model_or_gold_loaded':True,'remote_root':str(Path(plan['whitelist']['path']).parent),'actual_CPU_extraction_seconds':summary['seconds'],'self_check_not_independent_review':True}
(R/'completion.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'completion_sha256':sha(R/'completion.json'),**report}))
