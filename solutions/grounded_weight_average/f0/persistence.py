"""One incremental F0 final entry/backup; inherits actual F1 weights and fixed policy."""
import argparse,hashlib,importlib.util,io,json,os,sys,time,zipfile
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S.parent));from synthesize import ref,save
R=Path('/workspace/patientphex-runs/round2-20261023');D=R/'research/grounded-weight-average-v1/F0';B=D/'final-backup-v1';F1=R/'research/grounded-weight-average-v1/F1'
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
REL='research/grounded-weight-average/F0-final-backup-v1';P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';MOUNT='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
def build():
 B.mkdir(exist_ok=False);result=json.loads((D/'quality-v1/result.json').read_text());freeze=json.loads((D/'quality-v1/freeze.json').read_text())
 named={
 'native_CPU_collection':D/'native-cpu-v1/collection.json','native_CPU_result':D/'native-cpu-v1/terminal/actual/result.json',
 'inference_collection':D/'infer-v1/collection.json','inference_complete':D/'infer-v1/terminal/output/complete.json','inference_wrapper':D/'infer-v1/terminal/wrapper-complete.json','loaded_model_manifest':D/'infer-v1/terminal/output/run-manifest.json','byteexact_reconstruction':D/'infer-v1/direct-recovery/byteexact-reconstruction.json',
 'extraction':D/'quotes-v1/manifest.json','normalization_preparation':D/'normalization-preparation/manifest.json','normalization':D/'normalized-v1/manifest.json',
 'freeze':D/'quality-v1/freeze.json','evaluation_intent':D/'quality-v1/evaluation-intent.json','result':D/'quality-v1/result.json','format':D/'quality-v1/format.json','quality_bound_inputs':D/'quality-v1/bound-inputs.json','primary_prediction':D/'quality-v1/prediction.jsonl','unscored_pure_mean':D/'quality-v1/mean-new-only.jsonl',
 'F1_mean_and_query_implementation':F1/'final-review-entry.json','F1_backup':F1/'final-backup-v1/complete.json','F1_stable_review':R/'validation/grounded-weight-average-F1-final-review/review.json','ancestry':S.parent/'evidence/F0-ancestry.json','protocol':S/'execution-plan.json','normalization_amendment':S.parent/'normalization-amendment.json','actual_verifier':S/'verify_direct.py','quality_source':S/'postprocess.py'}
 if (D/'infer-v1/direct-recovery/complete-from-observation.json').exists():named['complete_observation_byte_recovery']=D/'infer-v1/direct-recovery/complete-from-observation.json'
 if (D/'infer-v1/direct-recovery/command-read/complete.json').exists():named['original_response_CPU_read_fallback']=D/'infer-v1/direct-recovery/command-read/complete.json'
 if (D/'quality-v1/metrics.json').exists():named['metrics']=D/'quality-v1/metrics.json'
 if (D/'Sap-CPU100-v1/collection.json').exists():
  for k,n in [('query_collection','collection.json'),('query_complete','terminal/output/complete.json'),('query_runtime','terminal/output/encoder-runtime.json'),('query_vector_locator','terminal/member-index.json')]:named[k]=D/'Sap-CPU100-v1'/n
 complete=json.loads((D/'infer-v1/terminal/output/complete.json').read_text());nm=json.loads((D/'normalized-v1/manifest.json').read_text())
 entry=dict(status='COMPLETE_FIXED_F0_JOINT_WEIGHT_MEAN_CPU_QUERY_CONTROL',refs={k:ref(v) for k,v in named.items()},result=result,coverage=dict(documents=16,patients=47,entities=947,raw_rows=47,EOS=complete['EOS_rows'],cap=complete['cap_rows'],anchors=nm['records'],necessary_queries=nm['necessary_public_queries']),unchanged_B4_relations=343,formula=freeze['formula'],mean_relations=freeze['mean'],intersection=freeze['intersection'],added=freeze['added'],new_v2=result['new_v2'],old_v2=0,no_originaldev_or_official=True,no_checkpoint_or_alpha_or_precision_selection=True,source=ref(__file__),provenance_notes=['Same actual rank32 alpha32 synthesized effective delta mean as F1; no fitting, epoch or weight selection.','Direct original responses +94 JSON serializations verified byteexact against original complete manifest; no remote directory/archive completeness claim.','Same joint CPU FP32 query/device implementation as F1, original public bank/aliases/policy unchanged; no FP16 equivalence or pure causal attribution claimed.'],model_and_embeddings_retained_remotely=True,entry_time_unix=time.time())
 save(D/'final-review-entry.json',entry);files={}
 for p in S.rglob('*'):
  if p.is_file() and p.suffix in ['.py','.json','.md'] and '__pycache__' not in p.parts:files['source/f0/'+p.relative_to(S).as_posix()]=ref(p)
 for folder in ['quality-v1','quotes-v1','normalization-preparation','normalized-v1','Sap-CPU100-v1/terminal','native-cpu-v1/terminal']:
  for p in (D/folder).rglob('*'):
   if p.is_file():files['actual/'+p.relative_to(D).as_posix()]=ref(p)
 for p in list(named.values())+[D/'final-review-entry.json',D/'infer-v1/terminal/output/responses.jsonl']:
  if D in p.parents:files['actual/'+p.relative_to(D).as_posix()]=ref(p)
 for n in ['normalization-amendment.json','persistence.py','synthesize.py','evidence/F0-ancestry.json']:files['source/'+n]=ref(S.parent/n)
 for n in ['f0/extract_quotes.py','f0/held_prepare.py','f0/held_finalize.py','schema.py','inference/decode.py','quality/assemble.py','quality/support.py','normalization/strict.py','normalization/semantic.py','normalization/public_bank.py','normalization/policy_v1.json']:files['inherited/'+n]=ref(G/n)
 manifest=dict(files=files,not_self_contained_large_dependencies=['F1 public base/parents/new mean adapter and actual old evidence','fixed HPO/train32/pure inputs/public Sap bank/model; new query vectors exact remote locator'],no_gold_file_in_bundle=True);raw=io.BytesIO()
 with zipfile.ZipFile(raw,'w',zipfile.ZIP_DEFLATED) as z:
  for name,item in sorted(files.items()):assert ref(item['path'])==item;z.writestr(name,Path(item['path']).read_bytes())
  z.writestr('manifest.json',json.dumps(manifest,indent=2)+'\n')
 with (B/'bundle.zip').open('xb') as h:h.write(raw.getvalue());h.flush();os.fsync(h.fileno())
 save(B/'plan.json',dict(archive=ref(B/'bundle.zip'),manifest=manifest,mount=MOUNT,remote=P+'/'+REL,source=ref(__file__),transfer_source=ref(S.parent/'persistence.py'),review_entry=ref(D/'final-review-entry.json')))
 print(json.dumps(dict(plan=ref(B/'plan.json'),bytes=len(raw.getvalue()),members=len(files)+1)))
def transfer():
 spec=importlib.util.spec_from_file_location('f1_backup_io',S.parent/'persistence.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 m.D=D;m.B=B;m.REL=REL;m.MOUNT=MOUNT;m.P=P;m.transfer()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['build','transfer']);a=ap.parse_args();build() if a.stage=='build' else transfer()
