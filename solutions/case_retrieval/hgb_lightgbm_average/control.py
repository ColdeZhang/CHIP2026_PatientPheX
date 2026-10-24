"""Fixed 1:1 strong-HGB / original-LightGBM probability control; zero fit."""
import argparse, hashlib, json, math, signal, sys, time
from datetime import datetime, timezone
from pathlib import Path

def now():return datetime.now(timezone.utc).isoformat()
def ref(p):
    p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(r):
    p=Path(r['path']);a=ref(p);assert all(a[k]==r[k] for k in ('path','bytes','sha256') if k in r);return p
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    import gzip
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else open(p)) as f:return [json.loads(line) for line in f if line.strip()]
def write(p,j):
    with Path(p).open('x') as f:json.dump(j,f,indent=2);f.write('\n')
def sources(p):
    for path,digest in p['sources'].items():assert ref(path)['sha256']==digest
def canonical(j):return json.dumps(j,ensure_ascii=False,sort_keys=True,separators=(',',':'))

def aligned(p):
    prep=read(checked(p['prepared_manifest']));seen=set();out=[];counts=[]
    for fold in p['folds']:
        f=fold['fold'];a=read(checked(fold['hgb_manifest']));b=read(checked(fold['lightgbm_manifest']))
        for key in ('prepared_manifest','old_prepared_manifest','old_fit_manifest','reference','training_origins',
                    'training_documents','input_documents','training_examples','training_positive','input_candidates',
                    'features','threshold','label_function_sha256','dependencies'):
            assert a[key]==b[key],(f,key)
        assert a['fold']==b['fold']==f and a['actual_iterations']==b['actual_iterations']==160
        assert a['prepared_manifest']==p['prepared_manifest'] and a['threshold']==.3
        assert a['source_sha256']==p['hgb_fit_source']['sha256'] and b['source_sha256']==p['lightgbm_fit_source']['sha256']
        assert a['outputs']['selector-scores.jsonl']==fold['hgb_scores'] and b['outputs']['selector-scores.jsonl']==fold['lightgbm_scores']
        inputs=rows(checked(prep['folds'][f]['input']));aa=rows(checked(fold['hgb_scores']));bb=rows(checked(fold['lightgbm_scores']))
        assert len(inputs)==len(aa)==len(bb)==16
        scored=[];n=0;digest=hashlib.sha256()
        for row,x,y in zip(inputs,aa,bb):
            did=row['pmc_id'];assert did==x['pmc_id']==y['pmc_id'] and did not in seen;seen.add(did)
            assert set(row['input'])=={'pmc_id','pmid','patient','full_text'}
            assert len(x['candidates'])==len(y['candidates'])==len(row['candidates'])
            cs=[]
            for old,new,candidate in zip(x['candidates'],y['candidates'],row['candidates']):
                left={k:v for k,v in old.items() if k!='selector_probability'};right={k:v for k,v in new.items() if k!='selector_probability'}
                assert canonical(left)==canonical(right)==canonical(candidate)  # includes bool/zero type distinctions
                u,v=old['selector_probability'],new['selector_probability']
                assert type(u) is float and type(v) is float and math.isfinite(u) and math.isfinite(v) and 0<=u<=1 and 0<=v<=1
                mean=.5*u+.5*v
                assert math.isfinite(mean) and 0<=mean<=1
                cs.append(dict(candidate,selector_probability=mean));digest.update(mean.hex().encode()+b'\n')
            n+=len(cs);scored.append(dict(pmc_id=did,candidates=cs))
        assert n==a['input_candidates']==prep['folds'][f]['new_input_rows']
        out.append((f,inputs,scored));counts.append(dict(fold=f,documents=16,candidates=n,ordered_mean_floathex_sha256=digest.hexdigest(),features=len(a['features'])))
    assert len(seen)==64
    return out,counts

def execute(plan_path,align_only=False,receipt=None):
    started=time.monotonic();signal.alarm(30);p=read(plan_path);sources(p)
    assert p['weights']==[.5,.5] and p['cut']==.3 and p['CPU_cap_seconds']==30
    if align_only:
        _,counts=aligned(p)
        result=dict(status='PASS_pure_input_literal_candidate_and_lineage_alignment',plan=ref(plan_path),source=ref(__file__),folds=counts,
                    no_gold_fit_model_predict_or_decode=True,seconds=time.monotonic()-started)
        write(receipt,result);print(json.dumps(result));return
    alignment=read(checked(p['alignment_receipt']));assert alignment['source']==ref(__file__) and alignment['status'].startswith('PASS')
    root=Path(p['output_dir']);root.mkdir(exist_ok=False);state=dict(status='running_fixed_average_no_fit',pid=__import__('os').getpid(),started_utc=now(),plan=ref(plan_path))
    write(root/'execution-state.json',state)
    try:
        data,counts=aligned(p);assert counts==alignment['folds']
        sys.path.insert(0,p['literal_source_directory']);import control as literal
        # This file runs as __main__; imported control is the frozen literal editor.
        assert ref(literal.__file__)['sha256']==p['sources'][str(Path(literal.__file__).resolve())]
        sys.path.insert(0,p['nested_source']);import decode_nested_selector as decoder
        _,allowed,_,_=decoder.prior.base.parse_hpo(checked(p['hpo']));allowed.discard('HP:0000118')
        baseline=read(checked(p['baseline_freeze']));freeze=[];predictions={};inputbyid={};diagnostics=[]
        for f,inputs,scored in data:
            oldref=baseline['folds'][f]['new_prediction'];lines=checked(oldref).read_bytes().decode('utf-8').splitlines(keepends=True)
            assert len(lines)==len(inputs)==len(scored)==16
            output=[];docs=[];changed=0
            for line,row,score in zip(lines,inputs,scored):
                before=json.loads(line);did=row['pmc_id'];assert did==before['pmc_id']==score['pmc_id']
                cs=score['candidates'];selected,diag=decoder.decode(cs,decoder.greedy(cs,.3),row['input'],allowed,.3)
                encoded=literal.replace_entities(line,selected);output.append(encoded);docs.append(json.loads(encoded));inputbyid[did]=row['input']
                oldkeys={decoder.key(x) for x in before['entities']};newkeys={decoder.key(x) for x in selected}
                changed+=before['entities']!=selected
                diagnostics.append(dict(fold=f,pmc_id=did,entity_added=len(newkeys-oldkeys),entity_removed=len(oldkeys-newkeys),**diag))
            path=root/f'fold{f}.average.nested.jsonl';path.write_bytes(''.join(output).encode())
            freeze.append(dict(fold=f,baseline=oldref,prediction=ref(path),hgb_scores=p['folds'][f]['hgb_scores'],lightgbm_scores=p['folds'][f]['lightgbm_scores'],
                               mean_values=counts[f],changed_documents=changed));predictions[f]=docs
        sources(p);write(root/'input-only-diagnostics.json',diagnostics)
        write(root/'complete-freeze.json',dict(frozen_at_utc=now(),plan=ref(plan_path),all_four_complete=True,folds=freeze,
                no_gold_read=True,all_nonentity_literal_bytes_preserved=True,diagnostics=ref(root/'input-only-diagnostics.json')))
        for f in freeze:checked(f['prediction'])
        write(root/'evaluation-start.json',dict(started_utc=now(),freeze=ref(root/'complete-freeze.json')))
        sys.path.insert(0,p['evaluation_source']);import evaluate;import validate_predictions as validator
        assert evaluate.VERSION=='patientphex-public-approx-v2'
        gold=rows(checked(p['train64_gold']));truth={x['pmc_id']:x for x in gold};assert len(truth)==len(gold)==64 and set(truth)==set(inputbyid)
        assert all({k:v for k,v in row.items() if k not in ('entities','association')}==inputbyid[did] for did,row in truth.items())
        hpo=validator.read_hpo(checked(p['hpo']));results=[];allpred={}
        for f in range(4):
            docs=predictions[f];labels=[truth[x['pmc_id']] for x in docs];old=read(checked(p['baseline_fold_metrics'][f]))
            metrics=evaluate.evaluate_report(labels,docs,'all','text');valid=validator.validate_predictions(labels,docs,hpo)
            assert valid['valid'] and valid['error_count']==valid['warning_count']==0 and metrics['counts']['relation']==old['counts']['relation']
            write(root/f'fold{f}.metrics.json',metrics);write(root/f'fold{f}.validation.json',valid)
            results.append(dict(fold=f,baseline=old['metrics'],average=metrics['metrics'],deltas={k:metrics['metrics'][k]-old['metrics'][k] for k in old['metrics']},counts_before=old['counts'],counts_after=metrics['counts']))
            allpred.update({x['pmc_id']:x for x in docs})
        pooled=evaluate.evaluate_report(gold,[allpred[x['pmc_id']] for x in gold],'all','text');old=read(checked(p['baseline_pooled_metrics']))
        assert pooled['counts']['relation']==old['counts']['relation'];write(root/'pooled.metrics.json',pooled)
        nondecreasing=sum(x['deltas']['Score']>=0 for x in results);delta={k:pooled['metrics'][k]-old['metrics'][k] for k in old['metrics']}
        result=dict(status='complete_unique_fixed_probability_average',finished_utc=now(),plan=ref(plan_path),source=ref(__file__),folds=results,
                    pooled_baseline=old['metrics'],pooled_average=pooled['metrics'],pooled_deltas=delta,nondecreasing_folds=nondecreasing,
                    keep=delta['Score']>0 and nondecreasing>=3,new_fit_calls=0,new_model_prediction_calls=0,new_v2_calls=5,old_v2_calls=0,
                    changed_documents=sum(x['changed_documents'] for x in freeze),entity_added=sum(x['entity_added'] for x in diagnostics),entity_removed=sum(x['entity_removed'] for x in diagnostics),
                    nonentity_literal_bytes_preserved=True,no_originaldev_or_official=True,seconds=time.monotonic()-started)
        assert result['seconds']<30;write(root/'result.json',result);state.update(status='complete',finished_utc=now(),result=ref(root/'result.json'))
        print(json.dumps({k:result[k] for k in ['pooled_deltas','nondecreasing_folds','keep','seconds','changed_documents','entity_added','entity_removed']}))
    except BaseException as exc:state.update(status='failed_preserved',failure=repr(exc),finished_utc=now());raise
    finally:(root/'execution-state.json').write_text(json.dumps(state,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--align-only',action='store_true');ap.add_argument('--receipt');a=ap.parse_args()
    execute(Path(a.plan).resolve(),a.align_only,a.receipt)
