"""Bind the one actual CPU success to the original d52 model, full39 twice."""
import hashlib,json
from pathlib import Path
from prepare import S,T,G,R,B,D,P,ref,ident,save
ROOT=P+'/grounded-sampled-consensus-v1/infer-F1-v1'
CPU=P+'/grounded-sampled-consensus-v1/native-cpu-v1'

def main():
 native=D/'native-cpu-v1';cp=json.loads((native/'collection.json').read_text());assert cp['CPU_reported_success'] and cp['all_actual_SHA_verified']
 for v in cp['files'].values():assert ref(v['path'])==v
 result=json.loads((native/'terminal/actual/result.json').read_text());assert result['status']=='PASS_NATIVE_COMPLETE39_QUOTE_SAMPLING_INPUT' and result['rows']==39 and result['sampling_contract_passed'] and result['input_tokens']==203588 and result['max_input_tokens']==10080
 original=json.loads((G/'inference/infer-plan.json').read_text());old=json.loads((G/'f0/infer-plan.json').read_text());refs={k:dict(v) for k,v in old['payload_sources'].items()}
 refs.update({'run.py':ref(T/'run.py'),'raw_store.py':ref(T/'raw_store.py'),'sampling.py':ref(T/'sampling.py'),'infer_driver.py':ref(T/'infer_driver.py'),'native-CPU-result.json':ref(native/'terminal/actual/result.json')})
 for name in ['held-prompts.jsonl','held-actual-ids.json.gz','held-audit.json']:refs[name]=original['payload_sources'][name]
 for v in refs.values():assert ref(v['path'])==v
 names={k:Path(v['path']).name for k,v in old['assets'].items()};names['sampling']='sampling.py'
 assets={k:dict(path=ROOT+'/'+name,**ident(refs[name])) for k,name in names.items()}
 cpuplan=json.loads((T/'cpu-plan.json').read_text());from_path=T/'sampling.py'
 import importlib.util
 spec=importlib.util.spec_from_file_location('_grounded_fixed_sampling',from_path);sampling=importlib.util.module_from_spec(spec);spec.loader.exec_module(sampling)
 reuse={'native-CPU-result.json':dict(path=CPU+'/actual/result.json',original_root=CPU,original_member='actual/result.json',**ident(refs['native-CPU-result.json']),actual_collection=ref(native/'collection.json'))}
 p={k:old[k] for k in ['python','model','gpu_uuid','environment','storage_parent','checkpoint_directory','adapter','original_training_checkpoint','training_runtime','inference_runtime']}
 p.update(schema='grounded-sample78-inference-v1',phase='infer',inference_authorized=True,authorization='Root explicit fixed two complete F1 quote streams2400+TERM30; no training, checkpoint choice, selective retries or official actions',budget_seconds=2400,authorized_budget_seconds=2400,grace_seconds=30,expected_rows=78,patients_per_seed=39,all_rows_fresh=True,max_new_tokens=1600,seeds=list(sampling.SEEDS),generation=sampling.GENERATION,presence_penalty=0,assets=assets,payload_sources=refs,files={ROOT+'/'+k:ident(v) for k,v in refs.items()},output_dir=ROOT+'/output',argv=[old['python'],'-u',ROOT+'/run.py','--plan',ROOT+'/infer-plan.json'],operations_source=ref(T/'remote.py'),collection_source=ref(T/'collect.py'),binding_source=ref(__file__),input_sources=dict(preparation=ref(D/'preparation.json'),native_CPU_collection=ref(native/'collection.json'),original_F1_plan=ref(G/'inference/infer-plan.json'),original_same_cache_F0_plan=ref(G/'f0/infer-plan.json'),formal_collection=old['input_sources']['formal_collection'],train_target=old['input_sources']['train_target'],copy_complete=old['input_sources']['copy_complete'],decision=ref(S/'decision.json'),budget_revision=ref(S/'decision-budget-v2.json')),reuse_packages=[ref(native/'package.json')],reuse_actual_artifacts=reuse,budget_basis={**json.loads((D/'preparation.json').read_text())['cost'],'original_cap_seconds':1800,'cap_seconds':2400,'prelaunch_revision':ref(S/'decision-budget-v2.json')},quality_policy=cpuplan['quality_policy'])
 save(T/'infer-plan.json',p);print(json.dumps(ref(T/'infer-plan.json')))

if __name__=='__main__':main()
