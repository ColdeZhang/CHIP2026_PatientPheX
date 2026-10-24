"""One authorized matched strong-N union old-P2 baseline; no existing arm rescoring."""
from pathlib import Path
import argparse,time
from common import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args();p=read(a.plan);out=Path(p['output']);out.mkdir(exist_ok=False);started=time.time();deadline=started+60
 assert p['source_sha256']==ref(__file__)['sha256']
 for k in ['strong_N','old_P','strong_metrics','old_P_metrics','prompts']:check(p[k])
 compatible(p['strong_N']['path'],p['old_P']['path'])
 n=pair_map(docs(p['strong_N']['path']));old=pair_map(docs(p['old_P']['path']));combined=union(n,old)
 prompts=docs(p['prompts']['path']);pool={(str(x['pmc_id']),str(pid),h) for x in prompts for pid in x['patient_ids'] for h in x['code_map'].values()}
 pairs={(d,pid,h) for d,v in combined.items() for pid,hs in v.items() for h in hs};assert pairs<=pool and len(prompts)==80
 pred=render(p['strong_N']['path'],combined,out/'pred.jsonl')
 save(out/'freeze.json',dict(plan=ref(a.plan),source=ref(__file__),prediction=pred,source_predictions=[p['strong_N'],p['old_P']],all_nonrelation_bytes_exact=True,rows=16,entities=1189,patients=39,pairs=len(pairs),gold_read=False,frozen_unix=time.time()))
 save(out/'evaluation-started.json',dict(freeze=ref(out/'freeze.json'),at_unix=time.time(),new_calls=1,old_calls=0))
 m=quality(p,out/'pred.jsonl',out,deadline)
 save(out/'result.json',dict(status='COMPLETE_MATCHED_TWO_EPOCH_BASELINE_ONLY',metrics=m['metrics'],counts=m['counts'],strong_N=read(p['strong_metrics']['path'])['metrics'],old_P2=read(p['old_P_metrics']['path'])['metrics'],delta_vs_N={k:v-read(p['strong_metrics']['path'])['metrics'][k] for k,v in m['metrics'].items()},prediction=pred,metrics_ref=ref(out/'metrics.json'),freeze=ref(out/'freeze.json'),new_quality_calls=1,old_quality_calls=0,seconds=time.time()-started,no_new_training_or_inference=True))
 print(read(out/'result.json'))
if __name__=='__main__':main()
