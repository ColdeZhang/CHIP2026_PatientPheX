"""Bind completed original two-epoch model to excluded-F0 complete native58."""
from pathlib import Path
import hashlib,json,tarfile
S=Path(__file__).resolve().parent;ROOT=S.parents[1];B=Path('/workspace/patientphex-runs/round2-20261023/research/patient-response-code-v1');Q=B/'F0-readiness-v1';D=B/'F0-full58-v1';C=B/'controller';F1=B/'F1-full80-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def rd(p):return json.loads(Path(p).read_text())
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def main():
 assert rd(Q/'ancestry.json')['status']=='PASS_COMPLETE_F0_CANDIDATE_ANCESTRY_EXCLUDES_HELD'
 cpu=rd(Q/'native-terminal/native-result.json');assert cpu['status']=='PASS_NATIVE_CPU_RESPONSE_CODE_F0_CENSUS_NO_MODEL' and not cpu['cuda_initialized'] and not cpu['held_gold_read'];assert rd(Q/'native-terminal/terminal.json')['returncode']==0
 stats=cpu['groups']['held']['arms']['response_code'];assert stats['rows']==58 and stats['max_tokens']<=32768 and stats['max_tokens']+1600<=cpu['context_tokens'];cap=max(14848,stats['proposed_512_rejection_cap'])
 old=rd(F1/'plan.json');assert old['adapter_files']['adapter_model.safetensors']=='2c777e88f86ed6f1aa620755f30ec176668578fe6349009470c4a58941cc0316';done=rd(C/'campaign-complete.json');assert done['status']=='TWO_COMPLETE_EPOCHS_32_STEPS_DURABLE_ACKS_VERIFIED'
 for e in [1,2]:
  a=C/f'epoch-{e:03d}-export-ack.json';ack=rd(a);assert sha(a)==done[f'epoch{e}_ack_sha256'] and ack['completed_epochs']==e and ack['global_step']==16*e and rd(C/f'epoch-{e:03d}-source-ack-copy.json')['sha256']==sha(a)
 assert old['adapter_files']['adapter_model.safetensors']==rd(C/'epoch-002-export-ack.json')['files']['adapter/adapter_model.safetensors']['sha256']
 audit=[x for x in rd(Q/'native-terminal/native-row-audit.json') if x['arm']=='response_code'];assert len(audit)==58 and cpu['groups']['held']['response_data_sha256']==sha(Q/'held.response-code.jsonl')
 D.mkdir(exist_ok=False);save(D/'row-audit.json',audit)
 sources={n:S/n for n in ['infer_full58.py','probe_runner.py','legacy_runner.py','launch.py','response_codec.py','model-source-manifest.json']};sources.update({'held.response-code.jsonl':Q/'held.response-code.jsonl','training-campaign-complete.json':C/'campaign-complete.json','training-epoch2-persistent-ACK.json':C/'epoch-002-export-ack.json'})
 for n,p in sources.items():(D/n).write_bytes(p.read_bytes())
 files={n:ref(D/n) for n in list(sources)+['row-audit.json']};plan={**old,'schema':'patient-response-code-same32-fulltextF0-58-native-v1','status':'COMPLETE_F0_ANCESTRY_AND_NATIVE_CPU_BOUND_NO_GPU','remote':'patient-response-code/F0-full58-v1','fresh_rows':58,'input_limit':cap,'native_census_limits':dict(total_tokens=stats['total_tokens'],max_tokens=stats['max_tokens'],input_cap=cap),'longest_first_index':stats['longest_indices'][0],'run_order':'One longest original row first as one of58, then original ascending length/rowID; no extra probe/cache','files':{n:{k:v for k,v in r.items() if k!='path'} for n,r in files.items()},'native_census':ref(Q/'native-terminal/native-result.json'),'ancestry':ref(Q/'ancestry.json'),'quality_policy':ref(ROOT/'f0/quality-policy.json'),'comparison_scope':'Existing normalN head48 vs codedhead32; shared upstream candidates48 excludeF0; actual candidate comparison, not uniformtrain32 isolated encoding.'}
 save(D/'plan.json',plan);files['plan.json']=ref(D/'plan.json')
 with tarfile.open(D/'deployment.tar.gz','x:gz') as t:
  for n in files:t.add(D/n,arcname=n,recursive=False)
 with tarfile.open(D/'deployment.tar.gz') as t:
  assert set(t.getnames())==set(files)
  for n,r in files.items():b=t.extractfile(n).read();assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
 save(D/'deployment-manifest.json',dict(archive=ref(D/'deployment.tar.gz'),files=files))
 save(D/'entry.json',dict(plan=ref(D/'plan.json'),operations=ref(S/'operations.py'),deployment=ref(D/'deployment-manifest.json'),quality_plan=ref(ROOT/'f0/quality-plan.json'),all58fresh=True,old_cache_reuse=False))
 print(json.dumps(dict(entry=ref(D/'entry.json'),native_limits=plan['native_census_limits'],longest_first_index=plan['longest_first_index'])))
if __name__=='__main__':main()
