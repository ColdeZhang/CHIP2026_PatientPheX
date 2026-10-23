#!/usr/bin/env python3
"""Two complete same-template slot arms freeze before one fixed F1 evaluation."""
from pathlib import Path
import argparse,json,hashlib,datetime,subprocess,math,gzip
import numpy as np
from sklearn.metrics import roc_auc_score

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),bytes=Path(p).stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l]
def save(p,x):
    with Path(p).open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def checked(x):assert sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes']
def command(argv,path):
    x=subprocess.run(argv,capture_output=True,text=True)
    save(path,dict(argv=argv,returncode=x.returncode,stdout=x.stdout,stderr=x.stderr));assert x.returncode==0,x.stderr

def freeze(b,plan):
    r=plan['evaluation_references'];complete=read(b/'complete.json');assert complete['status']=='both80_fixed_slot_forward_complete'
    assert complete['plan_sha256']==sha(b/'remote-plan.json') and complete['manifest_sha256']==sha(b/'run_manifest.json')
    assert complete['elapsed_seconds']<300 and set(complete['arms'])=={'base','adapter'}
    # No gold/label path is opened here, including no hash-only label reads.
    for x in r.values():checked(x)
    encoded=json.loads(gzip.decompress(Path(r['encoded']['path']).read_bytes()))
    prompts=rows(r['prompts']['path']);by={x['row_id']:x for x in prompts};assert len(by)==80
    entities={str(x['pmc_id']):x['entities'] for x in rows(r['entities']['path'])}
    outputs={};probabilities={}
    for arm in ['base','adapter']:
        c=complete['arms'][arm];rawpath=b/f'{arm}.responses.jsonl';ppath=b/f'{arm}.probabilities.jsonl'
        assert sha(rawpath)==c['responses_sha256'] and sha(ppath)==c['probabilities_sha256']
        raw={x['row_id']:x for x in rows(rawpath)};prob={x['row_id']:x for x in rows(ppath)}
        assert len(raw)==len(prob)==80 and set(raw)==set(prob)==set(by)
        flattened=[]
        for e in encoded:
            p=by[e['row_id']];v=prob[e['row_id']];answer=json.loads(raw[e['row_id']]['response'])
            keys=[[cid,pid] for cid in p['code_map'] for pid in p['patient_ids']]
            assert keys==e['keys']==v['keys'] and set(answer)==set(p['code_map'])
            assert not raw[e['row_id']]['hit_token_limit'] and not raw[e['row_id']]['generated']
            assert v['input_ids_sha256']==hashlib.sha256(json.dumps(e['input_ids']).encode()).hexdigest()
            assert len(v['probabilities'])==len(v['logits'])==len(keys)
            expected={cid:[] for cid in p['code_map']}
            for (cid,pid),pr,logits in zip(keys,v['probabilities'],v['logits']):
                assert len(logits)==2 and all(math.isfinite(x) for x in logits) and 0<=pr<=1
                z=max(logits);manual=math.exp(logits[1]-z)/sum(math.exp(x-z) for x in logits)
                assert abs(pr-manual)<2e-7
                if pr>=.5:expected[cid].append(pid)
                flattened.append(dict(row_id=p['row_id'],cid=cid,patient_id=pid,probability=pr))
            assert answer==expected
        assert len(flattened)==1557
        pred=b/f'{arm}.pred.jsonl'
        command([plan['python'],r['parser']['path'],'--input',r['input']['path'],'--entities',r['entities']['path'],
            '--prompts',r['prompts']['path'],'--responses',str(rawpath),'--output',str(pred),'--fallback','baseline'],b/f'{arm}.merge-command.json')
        manifest=read(pred.with_suffix('.manifest.json'))
        assert manifest['stats']=={'valid_candidate_decisions':587}
        docs=rows(pred);assert len(docs)==16 and sum(len(d['entities']) for d in docs)==1189
        assert all(d['entities']==entities[str(d['pmc_id'])] for d in docs)
        outputs[arm]=ref(pred);probabilities[arm]=flattened
    save(b/'paired-probabilities-frozen.json',dict(arms=probabilities,threshold=.5,complete=ref(b/'complete.json')))
    save(b/'paired-full-predictions-freeze.json',dict(status='both_slot_arms_complete_frozen_before_F1_labels',
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan=ref(b/'run-plan.json'),
        arms=outputs,probabilities=ref(b/'paired-probabilities-frozen.json'),complete=ref(b/'complete.json'),
        original_SFT32_reference=r['old_prediction'],no_new_labels_read=True,entity_and_text_assembly_same_source=True))

def evaluate(b,plan):
    r=plan['evaluation_references'];f=read(b/'paired-full-predictions-freeze.json')
    assert f['plan']==ref(b/'run-plan.json') and f['status']=='both_slot_arms_complete_frozen_before_F1_labels'
    for x in f['arms'].values():checked(x)
    checked(f['probabilities']);checked(f['complete']);checked(f['original_SFT32_reference'])
    save(b/'evaluation-started.json',dict(freeze=ref(b/'paired-full-predictions-freeze.json'),
        at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fixed_threshold=.5,no_search=True))
    # First new-control label/gold access follows both complete arms' probabilities and full predictions.
    checked(plan['labels_after_freeze']);checked(plan['gold_after_freeze'])
    labels={x['row_id']:json.loads(x['answer']) for x in rows(plan['labels_after_freeze']['path'])}
    oldraw={x['row_id']:json.loads(x['response']) for x in rows(r['old_raw']['path'])}
    pair=read(f['probabilities']['path'])['arms'];metrics={};diagnostics={}
    def counts(pred,y):
        tp=int(np.sum(pred&y));fp=int(np.sum(pred&~y));fn=int(np.sum(~pred&y))
        return dict(tp=tp,fp=fp,fn=fn,F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    for arm in ['base','adapter']:
        pred=f['arms'][arm]['path'];validation=b/f'{arm}.validation.json';mp=b/f'{arm}.metrics.json'
        command([plan['python'],r['validator']['path'],'--input',r['input']['path'],'--pred',pred,
            '--hpo',r['hpo']['path'],'--output',str(validation)],b/f'{arm}.validation-command.json')
        assert read(validation)['valid']
        command([plan['python'],r['evaluator']['path'],'--gold',plan['gold_after_freeze']['path'],'--pred',pred,
            '--body-scope','all','--no-id-doc','text','--output',str(mp)],b/f'{arm}.evaluation-command.json')
        metrics[arm]=read(mp)
        values=pair[arm];y=np.array([v['patient_id'] in labels[v['row_id']][v['cid']] for v in values],dtype=bool)
        old=np.array([v['patient_id'] in oldraw[v['row_id']][v['cid']] for v in values],dtype=bool)
        pr=np.array([v['probability'] for v in values]);selected=pr>=.5
        diagnostics[arm]=dict(pairs=len(y),positive=int(y.sum()),old_greedy=counts(old,y),new=counts(selected,y),
            Brier=float(np.mean((pr-y)**2)),AUC=float(roc_auc_score(y,pr)),old_Brier=float(np.mean((old.astype(float)-y)**2)),
            old_FN_recovered=int(np.sum(~old&y&selected)),old_TP_lost=int(np.sum(old&y&~selected)),
            new_FP=int(np.sum(~old&~y&selected)),old_FP_removed=int(np.sum(old&~y&~selected)))
    assert all(metrics['base']['metrics'][k]==metrics['adapter']['metrics'][k] for k in ['menF1','docF1'])
    cached=read(r['old_summary']['path'])['arms']['baseline']
    assert cached['prediction']==r['old_prediction']
    result=dict(status='complete_same_template_base_vs_fresh_slot_adapter_F1_control',arms=metrics,pair_diagnostics=diagnostics,
        cached_original_SFT32=cached,freeze=ref(b/'paired-full-predictions-freeze.json'),
        delta_adapter_minus_base={k:metrics['adapter']['metrics'][k]-metrics['base']['metrics'][k] for k in metrics['base']['metrics']},
        delta_adapter_minus_originalSFT32={k:metrics['adapter']['metrics'][k]-cached['metrics'][k] for k in cached['metrics']},
        raw_label_convention='Original F1 raw-HPO labels, no canonical remap; fullv2 literal all/text unchanged',
        original_F1_previously_used_for_development=True,no_original_dev_or_outer_gold=True,no_search=True,no_official=True)
    save(b/'summary.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','evaluate']);p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();b=a.directory.resolve();plan=read(b/'run-plan.json');assert sha(__file__)==plan['evaluator_sha256']
    globals()[a.phase](b,plan)
if __name__=='__main__':main()
