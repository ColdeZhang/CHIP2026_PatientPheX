"""Collect small cache evidence and perform input-only artifact checks; no labels."""
from pathlib import Path
import sys,json,hashlib,base64,zipfile,io,shlex,collections
R=Path(__file__).parent;T=R/'terminal';T.mkdir(exist_ok=True)
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();mount='476f81cf-09e7-445b-a7f0-d0e12b12e50f'
def call(n,a):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':mount,**a}});assert not r.get('isError'),r;return r['structuredContent']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=['train64-cache/run-state.json','train64-cache/native.stdout','train64-cache/native.stderr','train64-cache/provenance.stdout','train64-cache/provenance.stderr','native-train64-cache-cpu-preparation/preparation.json','native-train64-cache/manifest.json','native-train64-cache/preparation.json','native-train64-final-cache/manifest.json','native-train64-final-cache/canonical-entities.jsonl','native-train64-final-cache/source-provenance.jsonl','native-train64-final-cache/per-document-reuse.json']
paths.append('native-train64-cache/GPU-first-batch.json')
code="from pathlib import Path;import json,zipfile,hashlib;paths="+repr(paths)+";s=json.loads(Path('train64-cache/run-state.json').read_text());assert s['status']=='complete_cache_without_quality_evaluation';z=zipfile.ZipFile('train64-cache/terminal-small-evidence.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6);[z.write(p,p) for p in paths];z.close();p=Path('train64-cache/terminal-small-evidence.zip');print(json.dumps({'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'files':{n:{'sha256':hashlib.sha256(Path(n).read_bytes()).hexdigest(),'bytes':Path(n).stat().st_size} for n in paths}}))"
r=call('run_command',{'command':'../public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code),'cwd':'public-phenotagger-biobert','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':30});assert r['exit_code']==0,r;index=json.loads(r['stdout'])
# Whole archive is small; read_file binary transport uses base64 and source-file SHA.
chunks=[]
for offset in range(0,index['bytes'],262144):
 r=call('read_file',{'path':'public-phenotagger-biobert/train64-cache/terminal-small-evidence.zip','offset':offset,'limit':262144});assert r['encoding']=='base64';chunks.append(base64.b64decode(r['content']))
b=b''.join(chunks);assert len(b)==index['bytes'] and hashlib.sha256(b).hexdigest()==index['sha256']
with zipfile.ZipFile(io.BytesIO(b)) as z:
 assert set(z.namelist())==set(paths)
 for n in paths:
  data=z.read(n);assert hashlib.sha256(data).hexdigest()==index['files'][n]['sha256'];p=T/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
(R/'artifacts-transfer.json').write_text(json.dumps({'mount_id':mount,'remote_root':'public-phenotagger-biobert','local_root':str(T),'archive':index,'raw_large_retained_remote':True},indent=2)+'\n')
manifest_path=T/'native-train64-final-cache/manifest.json';manifest=json.loads(manifest_path.read_text());plan=json.loads((R/'plan.json').read_text());assert manifest['plan_sha256']==sha(R/'plan.json');assert manifest['model_weight_sha256']==plan['model_weight_sha256'];assert manifest['fixed_policy']==plan['policy'];assert manifest['input_sha256']==sha(R/'train64.input.jsonl')
for n,v in manifest['outputs'].items():assert sha(T/'native-train64-final-cache'/n)==v['sha256']
docs=[json.loads(x) for x in (R/'train64.input.jsonl').read_text().splitlines()];byid={d['pmc_id']:d for d in docs};assert len(byid)==64 and list(byid)==manifest['input_ids'] and all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs)
P=R.parent/'public-phenotagger-reuse';mapping=json.loads((P/'original-column-to-current-HPO.json').read_text());canon={x['original_label']:x['current_label'] for x in mapping['mapping']};dm=json.loads((P/'native-pipeline/dictionary-current-HPO.json').read_text());canon.update({x['original_id']:x['current_id'] for x in dm['mapping']});labels=(P/'sources/03-lable.vocab').read_text().strip().splitlines()
rows=[json.loads(x) for x in (T/'native-train64-final-cache/canonical-entities.jsonl').read_text().splitlines()];flags=[json.loads(x) for x in (T/'native-train64-final-cache/source-provenance.jsonl').read_text().splitlines()];key=lambda r:(r['pmc_id'],r['offset'],r['length'],r['hpo_id']);pool=collections.defaultdict(list)
for r in rows:
 d=byid[r['pmc_id']];p=d['full_text'][r['passage_index']];a,b=r['local_start'],r['local_end'];assert 0<=a<b<=len(p['text']);assert r['text']==p['text'][a:b] and r['offset']==p['offset']+a and r['length']==b-a;assert canon[r['raw_hpo_id']]==r['hpo_id'] and r['hpo_id'] is not None;pool[key(r)].append(r)
assert len(pool)==len(flags)==manifest['counts']['provenance_distinct_keys']
for f in flags:
 same=pool[key(f)];assert f['raw_hpo_ids']==sorted({r['raw_hpo_id'] for r in same});assert f['actual_hybrid_scores']==sorted({r['score'] for r in same});assert all(type(f[k]) is bool for k in ['raw_neural_match','raw_dictionary_match','abbrev_inferred']);assert any(f[k] for k in ['raw_neural_match','raw_dictionary_match','abbrev_inferred']);assert all(labels[i] in f['raw_hpo_ids'] for i in f['matched_neural_head_columns'])
per=json.loads((T/'native-train64-final-cache/per-document-reuse.json').read_text());lines=collections.defaultdict(list)
for line in (T/'native-train64-final-cache/canonical-entities.jsonl').read_text().splitlines(keepends=True):lines[json.loads(line)['pmc_id']].append(line)
for ident in byid:assert per[ident]['rows']==len(lines[ident]) and per[ident]['line_bytes_sha256']==hashlib.sha256(''.join(lines[ident]).encode()).hexdigest()
native_manifest=json.loads((T/'native-train64-cache/manifest.json').read_text());assert native_manifest['outputs']['canonical-entities.jsonl']['sha256']==sha(T/'native-train64-final-cache/canonical-entities.jsonl')
gate=json.loads((T/'native-train64-cache/GPU-first-batch.json').read_text());assert gate['valid'] and gate['rows']==128 and gate['classes']==17266 and gate['actual_tensor_device'].endswith('GPU:0');assert gate['weight_sha256']==plan['model_weight_sha256'];assert gate['native_variables']==201 and gate['model_parameters']==121587826;assert sha(T/'native-train64-cache/GPU-first-batch.json')==native_manifest['GPU_first_batch_validation_sha256']
state=json.loads((T/'train64-cache/run-state.json').read_text());assert state['gpu_returncode']==state['union_returncode']==0 and state['gpu_elapsed_seconds']<480 and state['union_elapsed_seconds']<120;assert state['final_manifest_sha256']==sha(manifest_path);assert state['native_manifest_sha256']==sha(T/'native-train64-cache/manifest.json');assert state['plan_sha256']==sha(R/'plan.json')
report={'status':'PASS_input_only_contract_and_transfer','no_quality_evaluation':True,'model_type':'biobert','first_real_GPU_batch_gate_verified':True,'input_sha256':manifest['input_sha256'],'manifest_sha256':sha(manifest_path),'source_sha256':sha(__file__),'plan_sha256':sha(R/'plan.json'),'documents':64,'canonical_rows':len(rows),'distinct_candidate_keys':len(pool),'all_coordinates_original_text_exact':True,'all_rawIDs_map_current_valid_branch':True,'native_canonical_all_row_bytes_exact':True,'all64_per_document_line_byte_sha_verified':True,'sourceflags_definitions':'actual raw acceptance/dictionary/pre-vs-post abbreviation from CPU union; local checks key/schema/column metadata, no guessing from confidence','provenance_true_counts':{k:sum(f[k] for f in flags) for k in ['raw_neural_match','raw_dictionary_match','abbrev_inferred']},'gpu_elapsed_seconds':state['gpu_elapsed_seconds'],'union_elapsed_seconds':state['union_elapsed_seconds'],'release_after_gpu':state['release_after_gpu'],'files':{n:sha(T/'native-train64-final-cache'/n) for n in ['manifest.json','canonical-entities.jsonl','source-provenance.jsonl','per-document-reuse.json']}}
(R/'input-contract-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
