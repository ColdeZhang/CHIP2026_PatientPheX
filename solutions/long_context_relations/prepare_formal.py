"""Freeze authorized formal source, unchanged probe model/data and one clock."""
import hashlib,io,json,zipfile
from pathlib import Path
S=Path(__file__).resolve().parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/long-context-relations-v1')
P='/home/zhangyuheng/projects/CHIP2026_PheX'
def ref(p):p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def main():
 cfg=json.loads((S/'runtime/execution-plan.json').read_text());pkg=json.loads((D/'probe-v1/package.json').read_text())
 obs=json.loads(sorted((D/'probe-v1').glob('observation-*.json'))[-1].read_text());actual=json.loads(obs['stdout']);cp=actual['probe-output/complete.json'];end=actual['job/ended.json']
 assert cp['status']=='PASS_FULL_CONTEXT_RELATION_ENGINEERING_DISCARD'and end['returncode']==0 and not end['timed_out']
 assert cp['parameters']==149416193 and cp['actual_adam_updates']==1 and cp['saved_tensors_compared']==563
 assert cp['formal30_cost']['worst_all960_FB_seconds']<2400
 for name in ('model.py','common.py'):assert(S/'formal'/name).read_bytes()==(S/'runtime'/name).read_bytes()
 root=pkg['remote'].rsplit('/',1)[0];formal=root+'/formal-v1';out=D/'formal-v1';out.mkdir(exist_ok=True)
 plan=dict(schema='long-context-relations-formal-v1',budget_seconds=2400,term_grace_seconds=30,model=cfg['model'],python=cfg['python'],gpu_uuid=cfg['gpu_uuid'],environment=cfg['environment'],
  cpu_receipt=pkg['remote']+'/cpu-complete.json',remote_data_source=pkg['remote']+'/data.json.gz',data=cfg['data'],
  source_files={p.name:ref(p)for p in sorted((S/'formal').glob('*.py'))},
  recipe=dict(epochs=30,documents=32,document_visits=960,candidate_pairs_per_epoch=4495,positive_pairs=727,accumulation=4,actual_Adam=240,encoder_lr=2e-5,head_lr=1e-4,warmup=24,seed=20261024,cut=0,primary='B4_PROTECTED_UNION_ONLY',new_v2_max=1,old_v2=0),
  fresh_parent_only=True,probe_state_reuse_forbidden=True,probe_proof=dict(observation=ref(sorted((D/'probe-v1').glob('observation-*.json'))[-1]),checkpoint_discarded=cp['checkpoint'],resident_peak=cp['peak_training_allocated_bytes'],cost=cp['formal30_cost']),
  admission=dict(H69_launch_TTL_seconds=300,coordinator='Wait ontology training-seed necessarySap realrelease, then freshGPU0gate; no formal start based on historical probe release'),
  formal_budget_scope='All30epochs+fullstateACK+all16docs17windows2874pairs share single2400 clock; no per-phase reset')
 (S/'formal/formal-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 members={p.name:p for p in sorted((S/'formal').iterdir())if p.is_file()};refs={n:ref(p)for n,p in members.items()};buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED)as z:
  for n,p in members.items():z.writestr(n,p.read_bytes())
 (out/'source.zip').write_bytes(buf.getvalue());package=dict(mount=pkg['mount'],relative='research/long-context-relations/formal-v1',remote=formal,plan=ref(S/'formal/formal-plan.json'),archive=ref(out/'source.zip'),members=refs)
 (out/'package.json').write_text(json.dumps(package,indent=2)+'\n')
 checks=dict(status='FORMAL_MODEL_COMMON_BYTEEXACT_TO_ACTUAL_PROBE',model=ref(S/'formal/model.py'),common=ref(S/'formal/common.py'),data=cfg['data'],actual_probe=cp['status'],source=ref(__file__),epochs=30,updates=240,held17windows_and2874pairs=True,no_new_model_or_labels=True)
 (out/'source-contract.json').write_text(json.dumps(checks,indent=2)+'\n')
 print(json.dumps(dict(plan=package['plan'],archive=package['archive'])))
if __name__=='__main__':main()
