"""Frozen F0-only CPU inputs from original full-report58; never reads held gold."""
from pathlib import Path
import gzip,hashlib,json,sys,tarfile,time
S=Path(__file__).resolve().parent;ROOT=S.parent;R=Path('/workspace/patientphex-runs/round2-20261023');B=R/'research/patient-response-code-v1/F0-readiness-v1';Q=R/'research/patient-response-code-readiness-20261024';OLD=R/'research/qwen35-sft32-full-report-feasibility/f0-preparation'
sys.path.insert(0,str(ROOT/'inference'));from response_codec import transform,TABLE_PREFIX
sha=lambda b:hashlib.sha256(b).hexdigest()
def ref(p):p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=sha(b))
def save(p,x):
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
 start=time.monotonic();B.mkdir(exist_ok=False);b=(OLD/'expanded58.jsonl').read_bytes();assert sha(b)=='d633225256dbf1cd5b64e94c74d7ce7f220b834cd748158082b306f12191fab2';rows=[json.loads(x) for x in b.splitlines()];assert len(rows)==58 and all('answer' not in r for r in rows)
 train={str(json.loads(x)['pmc_id']) for x in (Q/'train.response-code.jsonl').read_text().splitlines()};assert len(train)==32
 new=[];audit=[];maps={}
 for i,r in enumerate(rows):
  v=transform(r);new.append(v);u=r['messages'][1]['content'];assert v['messages'][1]['content'][:len(u)].encode()==u.encode() and v['messages'][1]['content'][len(u):].startswith(TABLE_PREFIX)
  assert {k:x for k,x in v.items() if k not in ['messages','response_code_map','response_patient_codes','response_representation']}=={k:x for k,x in r.items() if k!='messages'}
  d=str(r['pmc_id']);assert d not in maps or maps[d]==v['response_code_map'];maps[d]=v['response_code_map'];assert list(v['response_code_map'].values())==r['patient_ids']
  audit.append(dict(index=i,row_id=r['row_id'],pmc_id=d,original_user_utf8_sha256=sha(u.encode()),preserved_prefix_sha256=sha(v['messages'][1]['content'][:len(u)].encode()),clinical_PID_order_and_CID_metadata_exact=True,no_held_target=True))
 assert len(maps)==16 and not(set(maps)&train);assert sum(len(m) for m in maps.values())==47;assert sum(len(r['code_map']) for r in rows)==415;assert sum(len(r['code_map'])*len(r['patient_ids']) for r in rows)==1166
 (B/'held.response-code.jsonl').write_bytes(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new).encode());(B/'held.original.jsonl.gz').write_bytes(gzip.compress(b,mtime=0));save(B/'row-contract.json',audit)
 baseplan=json.loads((Q/'cpu-plan.json').read_text());group=dict(rows=58,documents=16,patients=47,CID_values=415,patient_concept_pairs=1166,source=ref(OLD/'expanded58.jsonl'),output=ref(B/'held.response-code.jsonl'),compressed_original=ref(B/'held.original.jsonl.gz'))
 for name,src in {'native_cpu.py':S/'cpu/native_cpu.py','response_codec.py':ROOT/'inference/response_codec.py'}.items():(B/name).write_bytes(src.read_bytes())
 cp=json.loads((R/'research/patient-response-code-v1/controller/controller-plan.json').read_text());env={**cp['env'],'CUDA_VISIBLE_DEVICES':''}
 plan=dict(CPU_only=True,training_authorized=False,GPU_authorized=False,groups={'held':group},files={n:ref(B/n) for n in ['native_cpu.py','response_codec.py','held.original.jsonl.gz']},tokenizer_files=baseplan['tokenizer_files'],runtime={**baseplan['runtime'],'torch':'2.6.0+cu124'},mount_id=cp['mount_id'],remote='patient-response-code/F0-cpu-v1',python=cp['python'],model=cp['model'],env=env,cap_seconds=120,term_grace_seconds=5,ancestry=[ref(OLD/'construction-summary.json'),ref(OLD/'source-plan.json'),ref(Q/'train.response-code.jsonl')],train32_holdout_overlap=[],source=ref(__file__))
 save(B/'plan.json',plan);save(B/'cpu-plan.json',plan)
 files={n:ref(B/n) for n in ['native_cpu.py','response_codec.py','held.original.jsonl.gz','plan.json','cpu-plan.json']}
 with tarfile.open(B/'deployment.tar.gz','x:gz') as t:
  for n in files:t.add(B/n,arcname=n,recursive=False)
 save(B/'deployment-manifest.json',dict(archive=ref(B/'deployment.tar.gz'),files=files))
 save(B/'structural-result.json',dict(status='PASS_F0_COMPLETE58_PURE_INPUT_PT_SAME_CODEC',group=group,source=ref(__file__),codec=ref(ROOT/'inference/response_codec.py'),train32_documents=32,held_documents=16,overlap=[],all_original_user_UTF8_prefix_exact=True,old_renderer_called=False,original_clinical_IDs_markers_CID_HPO_order_exact=True,held_gold_read=False,baseline_native_N=ref(OLD/'full58-v1/quality/diagnostic-fulltext-alone/metrics.json'),seconds=time.monotonic()-start));print(json.dumps(dict(structural=ref(B/'structural-result.json'),archive=ref(B/'deployment.tar.gz'))))
if __name__=='__main__':main()
