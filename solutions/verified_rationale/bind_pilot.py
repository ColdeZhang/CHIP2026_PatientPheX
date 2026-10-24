"""Assemble exact source/input/model bindings only; no launch admission or GPU."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path

def ref(p,remote=None):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=remote or str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--readiness',required=True);ap.add_argument('--old-r1',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();R=Path(a.readiness);oldR=Path(a.old_r1);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);W=Path(__file__).parent;remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/verified-rationale-v1/pilot-v1';model='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/models/DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6'
 cache=json.loads(json.loads((R/'A30-100-cache-v3.json').read_text())['structuredContent']['stdout']);curr={Path(x['path']).name:x for x in cache['entries']};old=json.loads((oldR/'full80-preparation/recovered/terminal/result.json').read_text());expected=json.loads((R/'R1-expected-files.json').read_text());assets=[]
 for e in expected['files']:
  c=curr[e['name']];p=old['asset_stat_before'][e['name']]
  assert (c['bytes'],c['mtime_ns'],c['inode'])==(p['bytes'],p['mtime_ns'],p['inode']) and p['full_SHA_source']==e['sha256'] and c['bytes']==e['bytes']
  assets.append(dict(name=e['name'],bytes=e['bytes'],sha256=e['sha256'],mtime_ns=c['mtime_ns'],inode=c['inode']))
 write(out/'asset-identity.json',dict(status='SAME_HOST_OLD_FULLSHA_AND_FRESH_STAT_MATCH',MSrevision=expected['revision'],files=assets,old_full80=ref(oldR/'full80-preparation/recovered/terminal/result.json'),new_stat=ref(R/'A30-100-cache-v3.json')))
 data={'inputs.jsonl.gz':R/'native-cpu-v1/inputs.jsonl.gz','encoded15.jsonl.gz':R/'native-cpu-v1/collected/encoded15.jsonl.gz','audit.json':R/'native-cpu-v1/collected/audit.json','selection.json':R/'train-census/hard-probe-selection.json','fixed-pilot-plan.json':W/'pilot-plan.json','host-revision.json':W/'host-revision.json'}
 for name,p in data.items():shutil.copyfile(p,out/name)
 ms=json.loads(json.loads((R/'A30-length-memory-source.json').read_text())['structuredContent']['stdout']);config=ms['config'];config['rope_scaling']['attention_factor']=1.0;write(out/'derived-config.json',config)
 for name in ('pilot.py','pilot_driver.py','evidence_contract.py'):shutil.copyfile(W/name,out/name)
 cpu=json.loads((R/'native-cpu-v1/result.json').read_text());encoded=[json.loads(l) for l in gzip.decompress((out/'encoded15.jsonl.gz').read_bytes()).splitlines()];rows=[json.loads(l) for l in gzip.decompress((out/'inputs.jsonl.gz').read_bytes()).splitlines()];audits=json.loads((out/'audit.json').read_text());assert len(rows)==len(encoded)==len(audits)==15
 assert hashlib.sha256(gzip.decompress((out/'inputs.jsonl.gz').read_bytes())).hexdigest()=='932fde5734abebbfeb118ac5f40e4ba0fd8d908ddd24f6203c3b6e4afa949aba'
 assert [(x['row_id'],len(x['input_ids'])) for x in encoded]==[(x['row_id'],x['input_tokens']) for x in audits]
 assert sum(x['input_tokens'] for x in audits)==82445 and max(x['input_tokens'] for x in audits)==9400
 fixed=json.loads((out/'fixed-pilot-plan.json').read_text());host=json.loads((out/'host-revision.json').read_text());assert host['original_plan']['sha256']==ref(out/'fixed-pilot-plan.json')['sha256']
 refs={k:ref(out/n,remote+'/'+n) for k,n in {'inputs_gz':'inputs.jsonl.gz','encoded_gz':'encoded15.jsonl.gz','audit':'audit.json','selection':'selection.json','parent_plan':'fixed-pilot-plan.json','host_revision':'host-revision.json','asset_identity':'asset-identity.json','derived_config':'derived-config.json'}.items()}
 plan=dict(status='SOURCE_BOUND_READY_ONLY_AFTER_OWNER_G47_AND_SAP_RELEASE_AND_FRESH_ADMISSION',source_of_binding=ref(__file__),parent_plan_sha256=refs['parent_plan']['sha256'],sources={n:ref(out/n)['sha256'] for n in ('pilot.py','pilot_driver.py','evidence_contract.py')},references=refs,model_dir=model,gpu_uuid=host['gpu_uuid'],python=host['python'],PYTHONPATH=host['PYTHONPATH'],fixed={'rows':15,'max_new_tokens':3072,'pilot_seconds':2700},row_order=fixed['generation']['row_order'],empirical_memory_proxy_bytes=23366319357,input_uncompressed_sha256='932fde5734abebbfeb118ac5f40e4ba0fd8d908ddd24f6203c3b6e4afa949aba',admission_required='New valid300s separateSHA-bound admission with exclusive launch.accepted fencing; no admission generated until priorowner complete release',runtime_source_refs={'generation/utils.py':ms['generation/utils.py']['sha256'],'models/qwen3/modeling_qwen3.py':ms['models/qwen3/modeling_qwen3.py']['sha256']},no_student=True,no_quality=True)
 write(out/'execution.json',plan);write(out/'binding.json',dict(status='PASS15_INPUTS_MODEL_ANCESTRY_SOURCE_AND_UNCHANGED_RECIPE_BOUND_NO_LAUNCH',execution=ref(out/'execution.json'),all_files={p.name:ref(p) for p in out.iterdir() if p.is_file()},no_GPU=True));print(json.dumps(ref(out/'execution.json')))
if __name__=='__main__':main()
