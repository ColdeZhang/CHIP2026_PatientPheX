#!/usr/bin/env python3
"""Unchanged original relation parser; full freeze before sole new evaluation."""
from pathlib import Path
import argparse,datetime,hashlib,json,signal,subprocess,sys,time

def ref(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_bytes().splitlines()]
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['assemble','evaluate']);ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--gold',type=Path);ap.add_argument('--scripts',type=Path);ap.add_argument('--hpo',type=Path);a=ap.parse_args()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU>30s')));signal.alarm(30)
    tick=time.monotonic();b=a.directory;p=load(b/'plan.json');refs=p['local_references']
    assert ref(__file__)['sha256']==p['control_sources']['finish.py']
    for r in refs.values():assert ref(r['path'])['sha256']==r['sha256']
    complete=load(b/'terminal/complete.json');assert complete['plan_sha256']==ref(b/'plan.json')['sha256']
    assert complete['responses_sha256']==ref(b/'terminal/responses.jsonl')['sha256']
    if a.mode=='assemble':
        assert a.gold is a.scripts is a.hpo is None
        assert ref(b/'legacy/merge_pair_rag.py')['sha256']==p['files']['legacy/merge_pair_rag.py']
        out=b/'assembled';out.mkdir(exist_ok=False)
        cmd=[sys.executable,str(b/'legacy/merge_pair_rag.py'),'--input',refs['input']['path'],
             '--entities',refs['entities']['path'],'--prompts',str(b/'prompts.jsonl'),
             '--responses',str(b/'terminal/responses.jsonl'),'--output',str(out/'pred.jsonl'),'--fallback','baseline']
        run=subprocess.run(cmd,text=True,capture_output=True,timeout=20)
        save(out/'command.json',dict(argv=cmd,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr));assert run.returncode==0
        pred,old=rows(out/'pred.jsonl'),rows(refs['baseline']['path'])
        assert [d['pmc_id'] for d in pred]==[d['pmc_id'] for d in old]
        for x,y in zip(pred,old):assert {k:v for k,v in x.items() if k!='association'}=={k:v for k,v in y.items() if k!='association'}
        save(out/'freeze.json',dict(frozen_at_utc=now(),plan=ref(b/'plan.json'),responses=ref(b/'terminal/responses.jsonl'),
            prediction=ref(out/'pred.jsonl'),baseline=refs['baseline'],parser_manifest=ref(out/'pred.manifest.json'),
            all_nonassociation_fields_exact=True,gold_read=False,seconds=time.monotonic()-tick))
        print((out/'freeze.json').read_text())
    else:
        fr=load(b/'assembled/freeze.json');assert fr['gold_read'] is False and fr['prediction']==ref(b/'assembled/pred.jsonl')
        out=b/'evaluation';out.mkdir(exist_ok=False)
        save(out/'before-gold.json',dict(time_utc=now(),freeze=ref(b/'assembled/freeze.json'),gold_read=False,
            evaluator=ref(a.scripts/'evaluate.py'),validator=ref(a.scripts/'validate_predictions.py')))
        sys.path.insert(0,str(a.scripts));import evaluate as ev;import validate_predictions as va
        assert ev.VERSION=='patientphex-public-approx-v2'
        inputs=va.load_jsonl(refs['input']['path']);pred=va.load_jsonl(b/'assembled/pred.jsonl');gold=va.load_jsonl(a.gold)
        assert {d['pmc_id'] for d in inputs}=={d['pmc_id'] for d in gold}
        hpo=va.read_hpo(a.hpo);assert hpo['version']=='hp/releases/2026-06-23'
        validation=va.validate_predictions(inputs,pred,hpo);save(out/'validation.json',validation);assert validation['valid']
        report=ev.evaluate_report(gold,pred,'all','text');save(out/'metrics.json',report)
        base=load(refs['baseline_metrics']['path']);delta={k:report['metrics'][k]-v for k,v in base['metrics'].items()}
        assert delta['menF1']==delta['docF1']==0
        keep=delta['micF1']+delta['macF1']>0 and delta['macF1']>=0
        save(out/'results.json',dict(completed_at_utc=now(),baseline_metrics=base['metrics'],new_metrics=report['metrics'],
            delta=delta,keep=keep,keep_rule='micro+macro strictly increases and macro does not decrease',baseline_reused=True,
            relation_count_delta={k:report['counts']['relation'][k]-base['counts']['relation'][k] for k in ['tp','fp','fn']},
            seconds=time.monotonic()-tick,scope='One developmentF1 input control; oldA800/newA100 runtime difference retained. No originaldev.'))
        print((out/'results.json').read_text())

if __name__=='__main__':main()
