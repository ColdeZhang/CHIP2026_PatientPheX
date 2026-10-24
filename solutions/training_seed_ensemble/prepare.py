"""Local stdlib readiness only: frozen Grounded32 recipe, one new train seed.

No model/runtime import, remote call, tokenizer, held gold, fit or scorer.
The emitted training plan stays disabled and no GPU launch entry is emitted.
"""
from pathlib import Path
import argparse, ast, copy, gzip, hashlib, json, random, re, sys, time

OLD_SEED=20261023
NEW_SEED=20261025
SOURCE_SHA='198d1de2234bff7d60d662f37ecc41bed7c25efd6ebedbe665c7f58705871e8e'
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation')
R=Path('/workspace/patientphex-runs/round2-20261023/research')

def ref(path):
 p=Path(path).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def load(path):return json.loads(Path(path).read_text())
def save(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def pairs(path):
 rows=[json.loads(s) for s in Path(path).read_text().splitlines()]; keys=set();patients=set()
 for d in rows:
  for a in d['association']:
   patients.add((d['pmc_id'],a['patient_id']))
   for h in a['phenotype']:
    assert isinstance(h,str)
    for x in h.split(';') if re.fullmatch(r'HP:\d{7}(?:;HP:\d{7})+',h) else [h]: keys.add((d['pmc_id'],a['patient_id'],x))
 return rows,patients,keys

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--support-contract',type=Path,default=Path(__file__).parent/'evidence/support-inputs.json');a=p.parse_args();out=a.output.resolve()
 assert Path(__file__).resolve().parents[2] not in out.parents;out.mkdir(parents=True,exist_ok=False)
 def no_gold(event,args):
  if event=='open' and isinstance(args[0],(str,bytes)) and '.gold.' in str(args[0]):raise PermissionError('No held gold in preparation')
 sys.addaudithook(no_gold);started=time.monotonic()
 trainer=G/'training/train.py';source=trainer.read_text();assert ref(trainer)['sha256']==SOURCE_SHA
 old=ast.parse(source);sites=[(n.lineno,n.col_offset) for n in ast.walk(old) if isinstance(n,ast.Constant) and n.value==OLD_SEED]
 assert len(sites)==3 and source.count(str(OLD_SEED))==3
 changed=source.replace(str(OLD_SEED),str(NEW_SEED));new=ast.parse(changed)
 class Undo(ast.NodeTransformer):
  def visit_Constant(self,node):
   if node.value==NEW_SEED:node.value=OLD_SEED
   return node
 assert ast.dump(old,include_attributes=False)==ast.dump(Undo().visit(copy.deepcopy(new)),include_attributes=False)
 compile(changed,'train.py','exec');(out/'train.py').write_text(changed)
 original_plan=load(G/'training/formal-plan.json');assert original_plan['configuration']['seed']==OLD_SEED
 actual=load(R/'grounded-generation-v1/formal-v1/terminal/output/run_manifest.json')
 assert actual['fingerprint']['configuration']==original_plan['configuration']
 assert actual['fingerprint']['source_sha256']==SOURCE_SHA and actual['examples']==87 and actual['total_optimizer_steps']==22
 data=Path(original_plan['payload_sources']['native-encoded.json.gz']['path']);assert ref(data)==original_plan['payload_sources']['native-encoded.json.gz']
 train=json.loads(gzip.decompress(data.read_bytes()))['train'];assert len(train)==87
 audit=load(original_plan['payload_sources']['train-audit.json']['path']);assert len(audit)==87
 # No target generation or use of held labels: retain the exact prior encoded artifact.
 assert sum(len(x['input_ids']) for x in train)==451672 and sum(x['supervised_tokens'] for x in train)==11884
 ids={x['pmc_id'] for x in audit};assert len(ids)==32
 groups={};shape=[]
 for seed in [OLD_SEED,NEW_SEED]:
  groups[str(seed)]={}
  for epoch in range(2):
   indices=list(range(87));random.Random(seed+epoch).shuffle(indices);gs=[indices[i:i+8] for i in range(0,87,8)]
   assert [len(g) for g in gs]==[8]*10+[7] and sorted(indices)==list(range(87))
   groups[str(seed)][str(epoch+1)]=gs
   for i,g in enumerate(gs):shape.append(dict(seed=seed,epoch=epoch+1,group=i+1,rows=len(g),tokens=sum(len(train[j]['input_ids']) for j in g),supervised=sum(train[j]['supervised_tokens'] for j in g),max_row_tokens=max(len(train[j]['input_ids']) for j in g)))
 oldg=load(R/'grounded-generation-v1/formal-v1/terminal/output/epoch-groups.json');assert oldg==groups[str(OLD_SEED)]
 assert all(groups[str(OLD_SEED)][str(e)]!=groups[str(NEW_SEED)][str(e)] for e in (1,2))
 save(out/'epoch-groups.json',groups);save(out/'group-shape-costs.json',shape)
 # The seed's RNG reach is established from the actual loader source, not a toy numerical model.
 loader=Path(original_plan['payload_sources']['slot_model.py']['path']);assert ref(loader)==original_plan['payload_sources']['slot_model.py']
 ls=loader.read_text()
 for token in ['random.seed(seed)','np.random.seed(seed)','torch.manual_seed(seed)','torch.cuda.manual_seed_all(seed)','protocol.attach_adapter(base, torch, peft)']:assert token in ls
 initializations=dict(source=ref(loader),Python_NumPy_Torch_CPU_CUDA=True,fresh_LoRA=True,frozen_pretrained_backbone=True,dropout_randomness=True,actual_new_LoRA_tensor_difference_measured=False)
 support_contract=load(a.support_contract)
 folds={}
 for fold,patient_count,score,g_path in [('F1',39,.701325377212172,R/'grounded-generation-v1/quality-F1-metadata-recovery-v1/assembled/new-only.jsonl'),('F0',47,.7281956289113009,R/'grounded-generation-v1/F0-pure-relations-v1/new-only.jsonl')]:
  base=R/'four-source-fusion-20261024'/f'{fold}-v1/prediction.jsonl';metrics=base.with_name('metrics.json');rows,bp,bk=pairs(base);gr,gp,gk=pairs(g_path)
  assert len(rows)==len(gr)==16 and bp==gp and len(bp)==patient_count
  assert load(metrics)['metrics']['Score']==score
  assert not ids & {d['pmc_id'] for d in rows},'train32 overlaps held document'
  for b,g in zip(rows,gr):assert b['pmc_id']==g['pmc_id'] and {k:v for k,v in b.items() if k not in ('entities','association')}=={k:v for k,v in g.items() if k not in ('entities','association')}
  sc=support_contract[fold]; support={}; counts={}; inherited=pairs(sc['inherited_baseline']['path'])[2]
  assert ref(sc['inherited_baseline']['path'])==sc['inherited_baseline'] and inherited<=bk
  assert ref(sc['source_receipt']['path'])==sc['source_receipt']
  for name in ['G','S','Q','C']:
   rr=sc[name];assert ref(rr['path'])==rr
   sr,sp,sk=pairs(rr['path']);assert [d['pmc_id'] for d in sr]==[d['pmc_id'] for d in rows] and sp==bp
   counts[name]=dict(stored_relations=len(sk),outside_B4=len(sk-bk))
   if name in sc['inherited_in']:
    assert inherited<=sk
    sk=sk-inherited
   support[name]=sk
  assert support['G']==gk
  universe=set().union(*support.values());assert len(universe-bk)==sc['expected_outside_B4']
  # Removing inherited B from stored S/Q cannot change any final B4 UNION result.
  folds[fold]=dict(base=ref(base),old_metrics=ref(metrics),old_G=ref(g_path),Score=score,documents=16,patients=patient_count,baseline_relations=len(bk),old_G_relations=len(gk),old_G_outside_baseline=len(gk-bk),support_sources={n:sc[n] for n in ['G','S','Q','C']},support_counts=counts,inherited_baseline=ref(sc['inherited_baseline']['path']),inherited_baseline_contained_B4=True,inherited_baseline_removed_from=sc['inherited_in'],support_union_outside_baseline=len(universe-bk),broader_only_outside_baseline=len((universe-bk)-gk),upper_bound_only_not_TP_or_score=True)
 # Candidate plan is intentionally not a launchable old namespace.
 plan=dict(schema='training-seed-ensemble-readiness-v2',GPU_authorized=False,train_seed=NEW_SEED,old_train_seed=OLD_SEED,source=ref(out/'train.py'),original_source=ref(trainer),original_plan=ref(G/'training/formal-plan.json'),configuration={**original_plan['configuration'],'seed':NEW_SEED},encoded=ref(data),initialization=initializations,train_documents=32,unchanged_train_and_target=True,folds=folds,formula=support_contract['formula'],support_contract=ref(a.support_contract),superseded_unexecuted_formula=support_contract['supersedes_unexecuted'],one_endpoint_per_fold=True,no_single_member_quality=True,future_new_v2_max_per_fold=1,future_old_v2=0,identity_endpoint_reuses_baseline_zero_v2=True,new_v2_now=0,old_v2_now=0,future_gate='F1 strict Score improvement before F0; both folds strict Score improvement required for full64 proposal, macro disclosed',normalization_policy=ref(G/'normalization/policy_v1.json'),new_sampling=False,greedy=True,max_new_tokens=1600,no_truncation=True,no_checkpoint_selection=True,normalization='Same train32 aliases/public41099 bank/b59 policy; new eligible query embeddings only; no fitting',fresh_seed_not_old_adapter_warmstart=True,requires_future_new_namespace_resource_binding=True)
 save(out/'proposal.json',plan)
 result=dict(status='PASS_LOCAL_SOURCE_SEED_AND_COMPLETE_GROUP_CONTRACT_NO_MODEL',seconds=time.monotonic()-started,seed_constant_sites=sites,AST_only_three_seed_values_changed=True,unchanged_loss_optimizer_schedule_shape=True,rows=87,train_docs=32,epochs=2,updates=22,warmup=2,tails=[7,7],tokens=451672,supervision=11884,max_tokens=9210,complete_coverage_both_epochs=True,old_groups_match_actual=True,new_groups_differ=True,initialization=initializations,formula_space=folds,new_model_calls=0,new_tokenizer_calls=0,new_remote_calls=0,new_gold_reads=0,new_quality_calls=0,new_training_calls=0,prepared_source=ref(out/'train.py'),proposal=ref(out/'proposal.json'))
 save(out/'result.json',result);print(json.dumps(dict(result=ref(out/'result.json'),proposal=ref(out/'proposal.json'),candidate_capacity={f:x['support_union_outside_baseline'] for f,x in folds.items()},seconds=result['seconds'])))
if __name__=='__main__':main()
