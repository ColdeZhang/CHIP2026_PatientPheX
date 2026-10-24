from pathlib import Path
import json,hashlib,datetime,math
V=Path(__file__).resolve().parent;S=V/'snapshots';R=V.parent.parent;E=R/'research/candidate-order-warmstart-entry'
def read(p):return json.loads(p.read_text())
def snap(rel):return S/rel.replace('/','--')
def get(rel):return read(snap(rel))
def ref(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
manifest=read(V/'snapshot-manifest.json');used={}
def verify(r):
 p=Path(r['path']);assert 'gold' not in p.name,p
 rel=str(p.relative_to(E)) if p.is_relative_to(E) else None
 if rel in manifest:p=Path(manifest[rel]['path'])
 got=ref(p);assert got['sha256']==r['sha256'] and got['bytes']==r['bytes'],(p,got,r)
 used[str(p)]=got;return p
Q='formal-ack-recovery/paired-evaluation/';F='formal-ack-recovery/'
plan=get(Q+'run-plan.json');summary=get(Q+'summary.json');freeze=get(Q+'paired-predictions-freeze.json');terminal=get(F+'terminal-result.json');started=get(Q+'evaluation-started.json')
for role,r in terminal.items():
 if isinstance(r,dict) and {'path','bytes','sha256'}<=r.keys():
  if role=='state':continue # mutable owner status is background only
  verify(r)
assert ref(snap('evaluate_paired.py'))['sha256']==plan['control_sha256']=='563da3a19527b28e96e07418eb2fee063762301faf9492630292d61615e8a79d'
assert ref(snap('prepare_recovered_evaluation.py'))['sha256']==plan['history_materializer_source_sha256']=='25f2669768d110497d974eddd4b768d52b18cfcaf19e603513e825b2901578fc'
for r in plan['execution_history'].values():verify(r)
assert datetime.datetime.fromisoformat(freeze['at_utc'])<datetime.datetime.fromisoformat(started['at_utc'])
verify(started['paired_freeze']);verify(freeze['plan']);assert started['one_v2_call_per_new_arm'] and started['old_metrics_cached_only']
base=verify(freeze['old']);base_docs=[json.loads(s) for s in base.read_text().splitlines()]
nonrel=lambda docs:{d['pmc_id']:{k:v for k,v in d.items() if k!='association'} for d in docs}
old_metrics=read(verify(summary['old_cached_metrics']))['metrics'];assert old_metrics==summary['old_metrics']
training=[];export_bytes=0;export_seconds=0;epoch_seconds=0;fingerprints={};arm_checks={}
for arm in ['original','reverse-first']:
 fingerprints[arm]=[]
 for ep in [1,2]:
  origin='formal/' if arm=='original' and ep==1 else F
  prefix=f'{origin}{arm}/';d=prefix+f'epoch-{ep:03d}/'
  cp=get(d+'checkpoint.json');ack=get(d+'export-ack.json');m=get(d+'run_manifest.json');prog=get(d+'progress.json')
  assert cp['completed_epochs']==ack['completed_epochs']==prog['completed_epochs']==ep
  assert cp['global_step']==ack['global_step']==prog['global_step']==16*ep
  assert ack['checkpoint_sha256']==ref(snap(d+'checkpoint.json'))['sha256']
  assert ack['run_manifest_sha256']==ref(snap(d+'run_manifest.json'))['sha256']
  assert ack['status']=='persistent_checkpoint_verified' and ack['fsync_all_files']
  assert ack['persistent_mount_id']=='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
  assert ack['persistent_directory']==f'case-retrieval/qwen35-preparation/order-warmstart-stage/checkpoints/{arm}/epoch-{ep:03d}'
  assert ack['source_checkpoint_directory']==f'/dev/shm/patientphex-order-warm32-1045-20261024/{arm}/epoch-{ep:03d}'
  assert sum(x['bytes'] for x in ack['files'].values())==ack['total_bytes']
  for n,h in cp['files'].items():assert ack['files'][n]['sha256']==h
  rb=get(prefix+f'export-{ep:03d}/persistent-actual-readback.json');assert rb['exit_code']==0 and json.loads(rb['stdout'])==ack
  assert m['fingerprint']==cp['fingerprint'];fp=cp['fingerprint'];fingerprints[arm].append(fp)
  assert fp['source_sha256']==plan['trainer_sha256']=='93301f6070f624672d8331b8fda7304089a2af44b95c83424513ef4895873ab4'
  assert fp['order_arm']==arm and fp['epoch_variants']==(['O','O'] if arm=='original' else ['R','O'])
  assert len(fp['training_document_ids'])==len(fp['excluded_document_ids'])==32 and not set(fp['training_document_ids'])&set(fp['excluded_document_ids'])
  assert fp['warm_start']['adapter_files']['adapter_model.safetensors']=='4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff'
  assert fp['warm_start']['old_optimizer_loaded']==fp['warm_start']['old_training_state_opened']==False
  assert (m['examples'],m['steps_per_epoch'],m['total_optimizer_steps'],m['warmup_steps'],m['supervised_tokens'])==(124,16,32,2,7170)
  assert not m['fresh_base_on_initial_run'] and m['fresh_optimizer_on_initial_run'] and not m['validation_labels_used']
  assert fp['environment']['modeling_source_sha256']=='b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3'
  assert fp['environment']['packages']['fla-core'] is None and fp['environment']['packages']['flash-linear-attention'] is None
  assert fp['base_files']['all15_files_verified']
  export_bytes+=ack['total_bytes'];export_seconds+=ack['seconds'];epoch_seconds+=prog['epoch_seconds']
  training.append({'arm':arm,'epoch':ep,'step':16*ep,'checkpoint':ref(snap(d+'checkpoint.json')),'ack':ref(snap(d+'export-ack.json')),'readback':ref(snap(prefix+f'export-{ep:03d}/persistent-actual-readback.json')),'epoch_seconds':prog['epoch_seconds']})
  process=get(prefix+f'train-{ep:03d}/process-terminal.json');assert not process['running'] and not process['remote_exit_code_available']
  assert '1792808951.5793016' in process['job']['command']
 assert fingerprints[arm][0]==fingerprints[arm][1]
 for role,r in plan['arms'][arm]['references'].items():verify(r)
 complete=get(F+arm+'/train-002/training_complete.json');assert complete['epochs']==2 and complete['global_step']==32 and complete['started_completed_epochs']==1
 assert complete['source_sha256']==plan['trainer_sha256'] and complete['total_parameters']==8997081600
 finalcp=get(F+arm+'/epoch-002/checkpoint.json');response_m=get(F+arm+'/responses.manifest.json')
 assert response_m['expected_rows']==80 and response_m['fingerprint']['thinking']==False
 assert response_m['fingerprint']['source_sha256']==plan['references']['runner']['sha256']
 assert response_m['fingerprint']['prompts_sha256']==plan['references']['prompts']['sha256']
 for n,h in response_m['fingerprint']['adapter_files'].items():assert finalcp['files']['adapter/'+n]==h==summary['arms'][arm]['adapter_files'][n]
 response_ref=plan['arms'][arm]['references']['responses'];raw=verify(response_ref);rows=[json.loads(s) for s in raw.read_text().splitlines()]
 assert len(rows)==len({x['row_id'] for x in rows})==80 and all(x['hit_token_limit']==False and x['removed_demonstrations']==0 for x in rows)
 assert response_m['output_sha256']==response_ref['sha256']
 for name,r in freeze['arms'][arm].items():
  if isinstance(r,dict) and {'path','bytes','sha256'}<=r.keys():verify(r)
 pred=Path(freeze['arms'][arm]['prediction']['path']);docs=[json.loads(s) for s in pred.read_text().splitlines()]
 assert len(docs)==16 and nonrel(docs)==nonrel(base_docs) and sum(len(d['entities']) for d in docs)==1189
 assert freeze['arms'][arm]['parse']=={'valid_candidate_decisions':587,'unresolved_candidate_decisions':0}
 v=read(verify(summary['arms'][arm]['format_file']));assert v['valid'] and v['error_count']==v['warning_count']==0
 metric=read(verify(summary['arms'][arm]['metrics_file']));assert metric['metrics']==summary['arms'][arm]['metrics']
 assert metric['evaluator_version']=='patientphex-public-approx-v2' and metric['configuration']['body_scope']=='all' and metric['configuration']['unlinked_document_mode']=='text'
 assert abs(metric['metrics']['Score']-sum(metric['metrics'][x] for x in ['menF1','docF1','micF1','macF1'])/4)<1e-14
 evalcmd=get(Q+arm+'/evaluation-command.json');assert evalcmd['returncode']==0
 arm_checks[arm]={'metrics':metric['metrics'],'relation_counts':metric['counts']['relation'],'rows':80,'valid_candidate_decisions':587,'unresolved_candidate_decisions':0,'nonrelation_fields_exact':True,'prediction':freeze['arms'][arm]['prediction'],'raw_response':response_ref,'format_errors':0,'format_warnings':0,'inference_seconds':response_m['elapsed_seconds']}
assert fingerprints['original'][0]['encoded_variant_sha256']==fingerprints['reverse-first'][0]['encoded_variant_sha256']
assert fingerprints['original'][0]['warm_start']==fingerprints['reverse-first'][0]['warm_start']
assert fingerprints['original'][0]['configuration']['fixed']==fingerprints['reverse-first'][0]['configuration']['fixed']
assert export_bytes==2079864798 and abs(export_seconds-233.85177051462233)<1e-8
assert abs(epoch_seconds-terminal['cost']['four_epochs_total_seconds'])<1e-8
finished=get(F+'all-responses-complete.json');oldstate=get('formal/execution-state-before-ack-recovery.json')
assert finished['seconds_from_original_start']<3900
release=get(F+'reverse-first/inference-gpu-release.json');assert release['exit_code']==0;rr=json.loads(release['stdout']);assert rr['empty'] and rr['gpu_uuid']=='GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc'
assert rr['server_epoch']<finished['original_deadline_unix']==1792808951.5793016
checks={'status':'PASS_no_gold_or_evaluation_rerun','new_source_and_history_refs_verified':len(used),'arms':arm_checks,'training':training,'all_four_full_state_persistent_bytes':export_bytes,'all_four_export_seconds':export_seconds,'four_epoch_seconds':epoch_seconds,'all_neural_seconds':finished['seconds_from_original_start'],'freeze_UTC':freeze['at_utc'],'first_gold_marker_UTC':started['at_utc'],'final_gpu_release':rr,'managed_exit_code':'Unavailable; terminal completion is established by full checkpoints/complete/response SHA and running=false, not invented rc0.','new_resume_numerical_equivalence_tested':False}
(V/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(ref(V/'checks.json')))
