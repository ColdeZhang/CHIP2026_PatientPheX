"""CPU-only verify actual same-BioBERT reused raw predictions, never load gold or model."""
import os,json,hashlib,time
from pathlib import Path
R=Path.cwd();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';start=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=json.loads((R/'dev16-cache/plan.json').read_text());source=R/'native-train64-cache';target=R/'native-dev16-cache';parent=json.loads((source/'manifest.json').read_text());native=json.loads((target/'manifest.json').read_text());assert parent['model_weight_sha256']==native['model_weight_sha256']==plan['model_weight_sha256'];assert parent['fixed_policy']==native['fixed_policy']==plan['policy']
assert parent['model_type']==native['model_type']==plan['model_type']=='biobert'
assert sha(source/'manifest.json')==plan['forward_reuse']['source_manifest_sha256']
assert sha(source/'unique-raw-predictions.jsonl')==plan['forward_reuse']['source_raw_sha256']
for d,m in [(source,parent),(target,native)]:assert sha(d/'unique-raw-predictions.jsonl')==m['outputs']['unique-raw-predictions.jsonl']['sha256']
old={}
for line in (source/'unique-raw-predictions.jsonl').open():
 r=json.loads(line);key=tuple(r['lower_tokens']);assert key not in old;old[key]=r
new=[];reused=[];seen=set()
for line in (target/'unique-raw-predictions.jsonl').open():
 r=json.loads(line);key=tuple(r['lower_tokens']);assert key not in seen;seen.add(key)
 if key in old:
  q=old[key];assert r['forward_origin']=='reused_same_BioBERT_train64_forward';assert r['parent_unique_index']==q['unique_index'] and r['top3']==q['top3'] and r['native_line']==q['native_line'];reused.append(r)
 else:assert r['forward_origin']=='new_public_forward';new.append(r)
assert len(old)==244938
assert len(reused)==4655 and len(new)==50548 and len(seen)==55203
fresh=[json.loads(x) for x in (target/'new-forward-predictions.jsonl').read_text().splitlines()];assert fresh==new
report={'status':'PASS_exact_same_model_forward_reuse','source_sha256':sha(__file__),'source_model_weight_sha256':plan['model_weight_sha256'],'source_manifest_sha256':sha(source/'manifest.json'),'target_manifest_sha256':sha(target/'manifest.json'),'source_raw_sha256':sha(source/'unique-raw-predictions.jsonl'),'target_raw_sha256':sha(target/'unique-raw-predictions.jsonl'),'new_forward_raw_sha256':sha(target/'new-forward-predictions.jsonl'),'source_keys':len(old),'target_keys':len(seen),'reused_forward_keys':len(reused),'new_forward_keys':len(new),'all_reused_top3_native_line_parent_index_exact':True,'all_new_rows_equal_actual_new_forward_file':True,'no_model_or_gold_loaded':True,'seconds':time.monotonic()-start}
(R/'dev16-cache/reuse-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
