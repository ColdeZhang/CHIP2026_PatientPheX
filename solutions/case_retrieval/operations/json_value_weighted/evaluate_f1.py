#!/usr/bin/env python3
"""One complete weighted124 original80 F1 arm; exact old JSON merger and all/text v2."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,subprocess,time
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def save(p,x):
 with Path(p).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def checked(r):assert ref(r['path'])==r,r['path']
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def command(args,p):
 r=subprocess.run(args,text=True,capture_output=True);save(p,dict(argv=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();b=a.directory;start=time.monotonic();plan=read(b/'run-plan.json');r=plan['references'];assert sha(__file__)==plan['control_sha256']
 for x in r.values():checked(x)
 h=module('weighted_tree',r['tree_helper']['path']);f=module('weighted_fusion',r['fusion_helper']['path']);h.verify_parser(r['parser']['path'])
 terminal=read(r['training_complete']['path']);cp=read(r['checkpoint']['path']);assert terminal['epochs']==2 and terminal['global_step']==32 and terminal['elapsed_seconds']<2100
 assert cp['completed_epochs']==2 and cp['global_step']==32 and not cp['partial_not_for_inference']
 rawfreeze=read(b/'responses-freeze.json');assert rawfreeze['plan_sha256']==sha(b/'run-plan.json') and rawfreeze['raw_sha256']==sha(b/'responses.jsonl') and rawfreeze['manifest_sha256']==sha(b/'responses.manifest.json')
 prompts=rows(r['prompts']['path']);raw=rows(b/'responses.jsonl');f.verify_sft_response(h,read(b/'responses.manifest.json'),raw,b/'responses.jsonl',r['prompts']['path'],plan['adapter_files'],read(r['model_source']['path']))
 pairs=h.pair_rows(prompts);assert len(prompts)==80 and len(pairs)==1557 and sum(len(x['code_map']) for x in prompts)==587
 flags,parse=h.llm_features(prompts,raw,pairs)
 path=b/'pred.jsonl';assert not path.exists()
 command([plan['python'],r['parser']['path'],'--input',r['input']['path'],'--entities',r['old_entities']['path'],'--prompts',r['prompts']['path'],'--responses',str(b/'responses.jsonl'),'--output',str(path),'--fallback','baseline'],b/'assembly-command.json')
 base=rows(r['old_baseline_pred']['path']);newpred=rows(path);assert len(newpred)==16 and sum(len(d['entities']) for d in newpred)==1189
 assert [{k:v for k,v in d.items() if k!='association'} for d in newpred]==[{k:v for k,v in d.items() if k!='association'} for d in base]
 freeze=dict(status='cached_old_and_new_complete_original80_predictions_frozen_before_unique_eval',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(b/'run-plan.json'),old=r['old_baseline_pred'],new=ref(path),raw_freeze=ref(b/'responses-freeze.json'),entities1189_exact=True,parse=parse,assembly_manifest=ref(b/'pred.manifest.json'))
 save(b/'paired-predictions-freeze.json',freeze);save(b/'evaluation-started.json',dict(at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze=ref(b/'paired-predictions-freeze.json'),only_new_arm=True,originaldev_F0_unused=True))
 assert sha(plan['gold_path'])==plan['historical_F1_gold_sha256']
 command([plan['python'],r['validator']['path'],'--input',r['input']['path'],'--pred',str(path),'--hpo',r['hpo']['path'],'--output',str(b/'validation.json')],b/'validation-command.json');assert read(b/'validation.json')['valid']
 command([plan['python'],r['evaluator']['path'],'--gold',plan['gold_path'],'--pred',str(path),'--body-scope','all','--no-id-doc','text','--output',str(b/'metrics.json')],b/'evaluation-command.json')
 old=read(r['old_baseline_metrics']['path']);new=read(b/'metrics.json');assert new['documents']==16 and new['patients']==39 and all(old['metrics'][k]==new['metrics'][k] for k in ['menF1','docF1'])
 checked(freeze['old']);checked(freeze['new'])
 result=dict(status='complete_original124_JSON_value_weight2_fixed_original80_F1',scope='Repeatedly observed training-side F1 development; no generalization/official claim; old A800 vs new A100 runtime difference disclosed',prediction=ref(path),freeze=ref(b/'paired-predictions-freeze.json'),metrics_file=ref(b/'metrics.json'),format=ref(b/'validation.json'),old_cached_metrics=r['old_baseline_metrics'],old_metrics=old['metrics'],new_metrics=new['metrics'],counts=new['counts'],delta={k:new['metrics'][k]-old['metrics'][k] for k in new['metrics']},keep_by_score_increase=new['metrics']['Score']>old['metrics']['Score'],original124_JSON_and80_prompts_unchanged=True,entities1189_exact=True,no_public62_no_search=True,elapsed_CPU_seconds=time.monotonic()-start)
 save(b/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
