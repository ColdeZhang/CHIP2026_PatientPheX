"""Record the already completed single F1 endpoint, without model or evaluator calls."""
import hashlib,json,os
from pathlib import Path
S=Path(__file__).resolve().parents[1]
D=Path('/workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1')
Q=D/'quality-F1-v1'
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def main():
 r=read(Q/'result.json');n=read(D/'normalized-v1/manifest.json');c=read(D/'infer-F1-A30-v1/collection.json');nc=read(D/'normalization-runtime-v1/collection.json');g=read(D/'infer-F1-A30-v1/terminal/output/complete.json')
 assert r['keep'] and r['new_v2']==1 and r['old_v2']==0 and r['metrics']['Score']==.7052260831087024
 names=['preparation.json','native-cpu-v1/collection.json','a30-host-inventory-v1.json','copy-A30-v1/copy-complete.json','copy-A30-v1/source-service-closed.json','infer-F1-A30-v1/collection.json','extraction-v1/manifest.json','normalization-preparation-v1/manifest.json','normalization-runtime-v1/collection.json','normalized-v1/manifest.json']
 names+=['quality-F1-v1/'+x for x in ['freeze.json','result.json','metrics.json','evaluation-intent.json','evaluation-launch.json','evaluation-terminal.json','prediction.jsonl','all-seed-keys.json','intersection-and-additions.json']]
 sources={str(p.relative_to(S)):ref(p) for p in sorted(S.rglob('*')) if p.is_file() and p.suffix in {'.py','.json'} and 'evidence' not in p.parts}
 entry=dict(schema='grounded-open-quote-two-sample-F1-review-entry-v1',status='POSITIVE_F1_PENDING_INDEPENDENT_REVIEW',refs={name:ref(D/name) for name in names},source_refs=sources,metrics=r['metrics'],baseline=r['baseline'],delta=r['delta'],added=14,added_TP=9,added_FP=5,new_v2=1,old_v2=0,
  actual_generation=dict(rows=78,documents=16,patients_per_seed=39,EOS=77,cap=1,whole_cap_row_abstained=True,wrapper_seconds=c['wrapper']['elapsed_seconds'],native_seconds=g['seconds'],max_allocated_bytes=g['max_gpu_allocated_bytes'],max_reserved_bytes=21921529856,output_tokens=20075),
  normalization=dict(necessary_queries=362,wrapper_seconds=nc['wrapper']['elapsed_seconds'],per_seed=n['per_seed']),
  no_single_seed_quality=True,no_other_arm_or_seed_scan=True,originaldev_gold_read=False,no_official_action=True,
  limitations=['Development F1, not blind or originaldev acceptance','Original train/F1 greedy A100 Torch2.6; new A30 Torch2.7.1 native, no cross-host bitwise or pure-sampling causality claim','One cap row abstained without retry; original39 patients retained'])
 target=D/'F1-complete-review-entry.json';save(target,entry)
 save(S/'evidence/F1-result.json',dict(entry=ref(target),result=ref(Q/'result.json'),prediction=ref(Q/'prediction.jsonl'),freeze=ref(Q/'freeze.json'),metrics=r['metrics'],baseline=r['baseline'],delta=r['delta'],added=14,added_TP=9,added_FP=5,keep_F1_candidate=True,independent_review='pending',originaldev_best_unchanged=True))
 print(json.dumps(ref(target)))
if __name__=='__main__':main()
