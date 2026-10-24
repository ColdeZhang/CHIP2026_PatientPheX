"""Package the one new full-context relation engineering; no model execution."""
import gzip,json,hashlib,io,zipfile
from pathlib import Path
S=Path(__file__).resolve().parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-relations-v1/probe-v1')
P='/home/zhangyuheng/projects/CHIP2026_PatientPheX'
B=Path('/workspace/patientphex-runs/round2-20261023/research/biomedical-long-context-readiness-20261024')
def ref(p):p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
 D.mkdir(parents=True,exist_ok=True);data={}
 for split in ('train','held'):
  data[split]=json.loads(gzip.decompress((B/'relation-readiness'/f'{split}-geometry.json.gz').read_bytes()))
  for row in data[split]:row['shape']=dict(max_encoder_tokens=max(len(w['ids'])for w in row['windows']),encoder_tokens=sum(len(w['ids'])for w in row['windows']),windows=len(row['windows']),pairs=len(row['pairs']),entities=len(row['entities']),physical_mentions=sum(len(e['spans'])for e in row['entities']))
 data['pareto']={name:max(range(32),key=lambda i:(data['train'][i]['shape'][key],data['train'][i]['pmc_id']))for name,key in [('longest','max_encoder_tokens'),('most_pairs','pairs')]}
 raw=json.dumps(data,separators=(',',':')).encode();(D/'data.json.gz').write_bytes(gzip.compress(raw,mtime=0))
 parent_cpu=ref(B/'probe-v1/collected/cpu-complete.json');parent_cpu['path']=P+'/research/biomedical-long-context/probe-v1/cpu-complete.json'
 parent_data=ref(B/'probe-v1/data.json.gz');parent_data['path']=P+'/research/biomedical-long-context/probe-v1/data.json.gz'
 plan=dict(schema='long-context-relations-probe-v1',budget_seconds=180,term_grace_seconds=30,
  model=P+'/research/biomedical-long-context/assets/bioclinical-modernbert-base-c3648aa8',
  python=P+'/sequence-crf/.venv/bin/python',gpu_uuid='GPU-07726469-bf7a-b832-9dee-019dc276dc1d',
  environment=dict(PYTHONPATH=P+'/semantic-neural/venv/lib/python3.10/site-packages',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),
  data=ref(D/'data.json.gz'),parent_cpu=parent_cpu,parent_native_data=parent_data,
  source_files={p.name:ref(p)for p in sorted((S/'runtime').glob('*.py'))},
  proposed_formal=dict(epochs=30,updates=240,accumulation=4,warmup=24,encoder_lr=2e-5,head_lr=1e-4,cut=0,seed=20261024,not_yet_authorized=True),
  same_candidate_pair_coverage=dict(train=4495,positive=727,total_train_gold=818,held=2874),
  selected_shapes={k:dict(pmc_id=data['train'][i]['pmc_id'],**data['train'][i]['shape'])for k,i in data['pareto'].items()},
  expected_parameters=149416193,probe=dict(actual_FB=6,actual_Adam=1,first_accumulation=4,resident_Adam_accumulation=2,full_state_reopen_restore=True,discard=True),
  H69_launch_TTL_seconds=300,no_held_model_or_quality=True)
 (S/'runtime/execution-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 members={p.name:p for p in sorted((S/'runtime').iterdir())if p.is_file()};members['data.json.gz']=D/'data.json.gz'
 refs={n:ref(p)for n,p in members.items()};buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED)as z:
  for n,p in members.items():z.writestr(n,p.read_bytes())
 (D/'package.zip').write_bytes(buf.getvalue());pkg=dict(mount='1593f159-19bc-41ca-8b4c-2e7455f5db99',relative='research/long-context-relations/probe-v1',remote=P+'/research/long-context-relations/probe-v1',model=plan['model'],plan=ref(S/'runtime/execution-plan.json'),archive=ref(D/'package.zip'),members=refs)
 (D/'package.json').write_text(json.dumps(pkg,indent=2)+'\n');print(json.dumps({'plan':pkg['plan'],'archive':pkg['archive'],'shapes':plan['selected_shapes']}))
if __name__=='__main__':main()
