import json
params=json.loads('{"remote": "/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/f0-full58-v1", "bundle_sha256": "94013fcdeebb386690046d62cba3054b3c5b2109df4f7c2623cea6bdf295ef75", "expected": {"inputs/all58-published-encoded.jsonl.gz": {"bytes": 75803, "sha256": "9a132bedb7edfa8817b8d7daecc1e1df5a96ec7f9a3b63bb8cd2bd4ed64adfcd"}, "inputs/prompts.jsonl.gz": {"bytes": 75241, "sha256": "d138963c8369b32a39a2ae15ed09693b886653a01928b2e5756568917e3af8e9"}, "full58_contract.py": {"bytes": 10023, "sha256": "44d91b4d6cccefe78a1d8f6bcaa579e6c9f386d002c159848ee89d5701a1bcb1"}, "infer_full58.py": {"bytes": 23161, "sha256": "bf5680a566cb94efc7951a9605e1848fc59540067d999c669bf271d5046cda8b"}, "launch_full58.py": {"bytes": 6354, "sha256": "c3927c531c9bdd5e303aa7aeadab2141f35f6127daf9e1295420747c61d7051f"}, "normalized_final.py": {"bytes": 4465, "sha256": "9c7cd604cae230f8a77bf0be3fd7edf2a3893e35c584de38edf8724c11c43efe"}, "raw_first.py": {"bytes": 2794, "sha256": "4b6a1c1a2673f3899b5f00ae0d12955cb6bcd2e20640c0f4dcdef77096f92281"}, "sparse_final.py": {"bytes": 2372, "sha256": "caf079e471450a5d873c7a2409ff457bf8f1c01f08e44867d7d10291898552d5"}, "strict_final.py": {"bytes": 2169, "sha256": "20943d8681b4cf81aee0465b0f217d3491460a311be00f3bbc2818b721013fad"}, "conditional-decision.json": {"bytes": 2768, "sha256": "96b30159a0217ba9bd1e2e6d56bf0e9122b0b912521368d70c1295d786b86e54"}, "plan.pending.json": {"bytes": 14052, "sha256": "e505ba5d8bb0ca1091fb9e69898d50ab9c47aa67dbe70f6e401a19739f6277a0"}}}')
from pathlib import Path
import os,hashlib,tarfile,io,sys,copy,time,shutil
root=Path(params['remote']);assert os.getuid()==root.stat().st_uid==1045;os.chmod(root,0o700);blob=(root/'deployment.tar.gz').read_bytes();assert hashlib.sha256(blob).hexdigest()==params['bundle_sha256'];expected=params['expected']
with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as tar:
 members=tar.getmembers();assert {m.name for m in members}==set(expected) and len(members)==len(expected)
 for member in members:
  assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
  data=tar.extractfile(member).read();e=expected[member.name];assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256'];path=root/member.name;path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():assert path.read_bytes()==data,'Never replace different deployed bytes'
  else:
   with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
for name,e in expected.items():
 path=root/name;assert path.stat().st_size==e['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==e['sha256']
 if name.endswith('.py'):compile(path.read_text(),name,'exec')
assert not (root/'wrapper-started.json').exists() and not (root/'terminal').exists() and not (root/'plan.json').exists()
plan=json.loads((root/'plan.pending.json').read_text());assert plan['execution_ready'] is False and plan['conditional_F1_gate']['status'].startswith('PENDING')
assert plan['process_budget_seconds']==6000 and plan['TERM_grace_seconds']==30 and plan['cached_indices']==[30] and len(plan['new_indices'])==57
sys.path.insert(0,str(root));from full58_contract import load_inputs
shadow=copy.deepcopy(plan);shadow['execution_ready']=True;start=time.monotonic();rows,encoded,audit,cached=load_inputs(shadow);seconds=time.monotonic()-start
assert 'torch' not in sys.modules and 'transformers' not in sys.modules and 'tokenizers' not in sys.modules
out={'status':'PASS_REMOTE_ACTUAL_58_INPUT_ONE_EXACT_CACHE_NO_GPU_PENDING_F1','uid':os.getuid(),'plan_sha256':hashlib.sha256((root/'plan.pending.json').read_bytes()).hexdigest(),'sources':plan['sources'],'all_deployment_files':expected,'rows':len(rows),'tokens':sum(len(e['input_ids']) for e in encoded),'cache_indices':sorted(cached),'cache_sha256':plan['cached_rows']['30']['sha256'],'program_seconds':seconds,'execution_ready_on_disk':False,'CPU_shadow_scope':'Only in-memory pure load_inputs receives execution_ready true; no launcher/native imports/model/gold/quality; original conditional plan remains false','disk_free_bytes':shutil.disk_usage(root).free,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'no_native_model_modules':True}
assert out['disk_free_bytes']>32*1024*1024
with (root/'actual-input-contract.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(out))
