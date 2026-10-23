#!/usr/bin/env python3
"""One post-freeze two-arm F1 evaluation; never select epochs or thresholds."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys,time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--gold',type=Path,required=True);ap.add_argument('--scripts',type=Path,required=True)
    ap.add_argument('--hpo',type=Path,required=True);a=ap.parse_args();b=a.directory;started=time.monotonic()
    plan=json.loads((b/'plan.json').read_text());freeze=json.loads((b/'assembled/freeze.json').read_text())
    assert freeze['gold_read'] is False and freeze['threshold']==.5 and freeze['plan_sha256']==sha(b/'plan.json')
    assert set(freeze['outputs'])=={'base','final'}
    for arm,item in freeze['outputs'].items():assert sha(item['path'])==item['sha256']
    assert sha(b/'terminal-infer/inference/complete.json')==freeze['inference_complete_sha256']
    result=b/'evaluation';result.mkdir(exist_ok=False)
    save(result/'before-gold.json',dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        freeze_sha256=sha(b/'assembled/freeze.json'),source_sha256=sha(__file__),gold_read=False,
        evaluator_sha256=sha(a.scripts/'evaluate.py'),validator_sha256=sha(a.scripts/'validate_predictions.py'),
        arms={arm:item['sha256'] for arm,item in freeze['outputs'].items()}))
    sys.path.insert(0,str(a.scripts));import evaluate as evaluator;import validate_predictions as validator
    assert evaluator.VERSION=='patientphex-public-approx-v2'
    inputs=validator.load_jsonl(plan['assembly']['references']['input']['path'])
    gold=validator.load_jsonl(a.gold);assert {x['pmc_id'] for x in gold}=={x['pmc_id'] for x in inputs}
    hpo=validator.read_hpo(a.hpo);assert hpo['version']=='hp/releases/2026-06-23'
    reports={}
    for arm,item in freeze['outputs'].items():
        pred=validator.load_jsonl(item['path'])
        validation=validator.validate_predictions(inputs,pred,hpo)
        save(result/(arm+'.validation.json'),validation);assert validation['valid'],validation
        report=evaluator.evaluate_report(gold,pred,'all','text')
        report.update(gold_sha256=sha(a.gold),prediction_sha256=item['sha256'])
        save(result/(arm+'.metrics.json'),report);reports[arm]=report['metrics']
    output=dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-started,
        metrics=reports,prediction_freeze_sha256=sha(b/'assembled/freeze.json'),
        before_gold_sha256=sha(result/'before-gold.json'),source_sha256=sha(__file__),
        limitations=['One fixed F1 development control; train32 gate is resubstitution selection.',
            'Both arms share legacy77 gate candidate family, not current97-feature best pipeline.',
            'All nonentity/relations preserved from old baseline; quality comparison is Task1 only.',
            'Only final epoch2 and cutoff.5; no additional selection or originaldev.'])
    save(result/'results.json',output);print(json.dumps(output,ensure_ascii=False))

if __name__=='__main__':main()
