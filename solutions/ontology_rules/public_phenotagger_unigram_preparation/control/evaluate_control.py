"""One fixed entity-only unigram append against the fourfold nested baseline."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,math,os
from pathlib import Path
import signal,sys,time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return list(map(json.loads,Path(path).read_text().splitlines()))
def now():return datetime.now(timezone.utc).isoformat()
def ref(path):return dict(path=str(path),sha256=sha(path),bytes=Path(path).stat().st_size)
def check(spec):assert sha(spec['path'])==spec['sha256'] and Path(spec['path']).stat().st_size==spec['bytes'],spec['path']
def dump(path,obj):
    with Path(path).open('x') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic();signal.alarm(58)
    p=json.loads(Path(a.plan).read_text());assert p['status']=='frozen_after_complete_input_only_cache_before_quality'
    assert p['CPU_cap_seconds']==60 and p['entity_threshold']==.95 and p['baseline_kind']=='audited_selector_nested_fourfold'
    assert sha(__file__)==p['source_sha256'] and os.environ['CUDA_VISIBLE_DEVICES']==''
    for spec in p['sources'].values():check(spec)
    for spec in p['inputs'].values():check(spec)
    for fold in p['folds']:
        for name in ('input','baseline','baseline_metrics'):check(fold[name])
    for spec in p['cache'].values():check(spec)
    cache=json.loads(Path(p['cache']['summary']['path']).read_text());state=json.loads(Path(p['cache']['state']['path']).read_text())
    cacheplan=json.loads(Path(p['inputs']['full_cache_plan']['path']).read_text())
    assert state['status']=='complete' and state['returncode']==0
    assert state['plan_sha256']==cache['plan_sha256']==p['inputs']['full_cache_plan']['sha256']
    assert state['summary_sha256']==p['cache']['summary']['sha256']
    assert cache['status']=='complete_all13145_unigram_input_only_cache'
    assert cache['counts']['unique_inputs']==13145 and cache['counts']['new_forward']==12889
    assert cache['counts']['reused_CPU_probe']==256 and cache['counts']['coordinates']==121413
    assert cache['source_sha256']==cacheplan['source_sha256'] and cache['policy']==cacheplan['policy']
    assert cache['outputs']['unigram-sidecar.jsonl']['sha256']==p['cache']['sidecar']['sha256']
    assert cache['outputs']['unigram-sidecar.jsonl']['bytes']==p['cache']['sidecar']['bytes']
    pure=read(p['inputs']['pure64']['path']);docs={d['pmc_id']:d for d in pure};assert len(docs)==64
    assert all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in pure)
    candidates=defaultdict(list);seen=set()
    for r in read(p['cache']['sidecar']['path']):
        assert r['pmc_id'] in docs and len(r['lower_tokens'])==1
        assert r['source']=='Bioformer_public_native_unigram_extension' and not r['dictionary_added'] and not r['abbreviation_added']
        assert r['raw_hpo_id']!='HP:None' and float(r['native_probability_5dp'])>.95
        assert r['score']==float(r['native_probability_5dp']) and math.isfinite(r['raw_probability'])
        k=(r['pmc_id'],r['offset'],r['length'],r['hpo_id']);assert k not in seen;seen.add(k)
        candidates[r['pmc_id']].append(r)
    assert len(seen)==cache['counts']['accepted_coordinates']
    dependencies=Path(p['sources']['assembler']['path']).parent;sys.path.insert(0,str(dependencies))
    import assemble_public_phenotagger as core
    import run as baseline
    import validate_predictions as validator
    import evaluate as ev
    assert sha(core.__file__)==p['sources']['assembler']['sha256']
    assert sha(baseline.__file__)==p['sources']['baseline']['sha256']
    assert sha(ev.__file__)==p['sources']['evaluator']['sha256'] and ev.VERSION=='patientphex-public-approx-v2'
    assert sha(validator.__file__)==p['sources']['validator']['sha256']
    allowed=baseline.load_hpo(p['inputs']['hpo']['path'])[1]
    hpo=validator.read_hpo(p['inputs']['hpo']['path'])
    out=Path(p['output_dir']);out.mkdir(exist_ok=False)
    allids=set();prepared=[];allcounts=Counter();fullnew=[]
    for f in p['folds']:
        inputrows=read(f['input']['path']);base=read(f['baseline']['path']);indexed={d['pmc_id']:d for d in base}
        ids={d['pmc_id'] for d in inputrows};assert len(ids)==len(indexed)==16 and ids==set(indexed) and not ids&allids
        allids|=ids;pred=[];counts=Counter()
        for d in inputrows:
            assert docs[d['pmc_id']]==d
            output,cnt=core.assemble_document(d,indexed[d['pmc_id']],candidates[d['pmc_id']],allowed)
            assert all(output[k]==v for k,v in indexed[d['pmc_id']].items() if k!='entities')
            assert Counter(json.dumps(e,sort_keys=True) for e in indexed[d['pmc_id']]['entities'])<=Counter(json.dumps(e,sort_keys=True) for e in output['entities'])
            pred.append(output);counts.update(cnt)
        dest=out/f"fold{f['fold']}";dest.mkdir()
        predpath=dest/'prediction.jsonl';predpath.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in pred))
        validation=validator.validate_predictions(inputrows,pred,hpo)
        assert validation['valid'] and validation['error_count']==0 and validation['warning_count']==0
        dump(dest/'validation.json',validation)
        info=dict(fold=f['fold'],input=f['input'],baseline=f['baseline'],prediction=ref(predpath),counts=dict(counts),validation=ref(dest/'validation.json'))
        dump(dest/'assembly-manifest.json',dict(info,gold_read=False,baseline_entities_all_preserved=True,all_nonentities_exact=True))
        prepared.append(info);fullnew+=pred;allcounts.update(counts)
    assert allids==set(docs) and len(fullnew)==64
    assert allcounts['native_candidates']==len(seen)
    pooledpath=out/'pooled.prediction.jsonl';pooledpath.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in fullnew))
    freeze=dict(status='all_four_complete_predictions_and_fixed_baselines_frozen_before_new_quality',frozen_at_utc=now(),source_sha256=sha(__file__),
        plan_sha256=sha(a.plan),cache_summary_sha256=p['cache']['summary']['sha256'],folds=prepared,pooled_prediction=ref(pooledpath),gold_read=False)
    dump(out/'complete-freeze.json',freeze)
    # First access to any task labels is strictly after all four new complete predictions.
    evaluation_started=now();check(p['gold_after_freeze']);gold=read(p['gold_after_freeze']['path']);goldmap={d['pmc_id']:d for d in gold}
    assert set(goldmap)==allids and len(gold)==64
    results=[]
    for f,info in zip(p['folds'],prepared):
        pred=read(info['prediction']['path']);g=[goldmap[d['pmc_id']] for d in pred]
        m=ev.evaluate_report(g,pred);m['prediction_sha256']=info['prediction']['sha256'];m['train64_gold_sha256']=p['gold_after_freeze']['sha256']
        old=json.loads(Path(f['baseline_metrics']['path']).read_text())
        assert m['metrics']['micF1']==old['metrics']['micF1'] and m['metrics']['macF1']==old['metrics']['macF1']
        dest=out/f"fold{f['fold']}"/'metrics.json';dump(dest,m)
        delta={k:m['metrics'][k]-old['metrics'][k] for k in m['metrics']}
        results.append(dict(fold=f['fold'],baseline=old['metrics'],new=m['metrics'],delta=delta,entity_objective_delta=(delta['menF1']+delta['docF1'])/2,
            counts_old=old['counts'],counts_new=m['counts'],assembly_counts=info['counts'],metrics=ref(dest)))
    pooled=ev.evaluate_report(gold,fullnew);dump(out/'pooled.metrics.json',pooled)
    old=json.loads(Path(p['inputs']['baseline_pooled_metrics']['path']).read_text())
    assert pooled['metrics']['micF1']==old['metrics']['micF1'] and pooled['metrics']['macF1']==old['metrics']['macF1']
    delta={k:pooled['metrics'][k]-old['metrics'][k] for k in pooled['metrics']}
    obj=(delta['menF1']+delta['docF1'])/2;nondecrease=sum(r['entity_objective_delta']>=0 for r in results)
    result=dict(status='complete_unique_fixed_unigram_append_quality',evaluation_started_at_utc=evaluation_started,completed_at_utc=now(),
        source_sha256=sha(__file__),plan_sha256=sha(a.plan),freeze_sha256=sha(out/'complete-freeze.json'),baseline_kind=p['baseline_kind'],
        folds=results,pooled=dict(baseline=old['metrics'],new=pooled['metrics'],delta=delta,entity_objective_delta=obj,counts_old=old['counts'],counts_new=pooled['counts']),
        retention=dict(pooled_entity_objective_strictly_improved=obj>0,nondecreasing_folds=nondecrease,retain=obj>0 and nondecrease>=3),
        counts=dict(allcounts),old_baseline_metrics_reused_without_reevaluation=True,all_old_entities_including_nested_preserved=True,
        all_nonentities_exact=True,format_errors=0,format_warnings=0,no_new_fit=True,no_originaldev=True,seconds=time.monotonic()-start,
        limitations=['Training-side fourfold development comparison, not untouched generalization or originaldev attainment.','Unique baseline is retained selector-nested-control, not old cut3 or alternate prior fits.','No retuning threshold, word list, model, gate, acceptance policy or baseline after quality.'])
    assert result['seconds']<60;dump(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
