"""Frozen F1 paired endpoints. Input-only construction precedes both gold reads."""
from pathlib import Path
from collections import defaultdict
import argparse, ast, copy, gzip, hashlib, importlib.util, json, math, os, subprocess, sys, time
import pool_helpers as ph

S=Path(__file__).parent
def ref(p): return ph.ref(p)
def save(p,x):
    with Path(p).open('x') as f: json.dump(x,f,indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
def load_module(name,p):
    spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',required=True); ap.add_argument('--mode',choices=['regression','prepare','evaluate'],required=True); ap.add_argument('--native-dir'); a=ap.parse_args()
    start=time.monotonic(); p=json.loads(Path(a.plan).read_text()); D=Path(p['output_directory']); D.mkdir(exist_ok=True)
    for r in p['fixed_inputs'].values():ph.checked(r)
    for r in p['sources'].values():ph.checked(r)
    assert ref(__file__)['sha256']==p['sources']['control']['sha256']
    if a.mode=='evaluate':
        freeze=json.loads((D/'paired-freeze.json').read_text()); assert freeze['gold_read'] is False
        for r in freeze['predictions'].values():ph.checked(r)
        save(D/'evaluation-intent.json',dict(at_epoch=time.time(),freeze=ref(D/'paired-freeze.json'),new_calls=2,old_calls=0))
        results={}
        for arm in ['primary','diagnostic']:
            target=D/(arm+'.metrics.json'); before=json.loads(Path(p['old_metrics'][arm]['path']).read_text()); ph.checked(p['old_metrics'][arm])
            assert time.time()<p['deadline_epoch'], 'Original campaign deadline expired before quality subprocess'
            argv=[p['python'],p['evaluator']['path'],'--gold',p['held_gold'],'--pred',freeze['predictions'][arm]['path'],'--body-scope','all','--no-id-doc','text','--output',str(target)]
            with (D/(arm+'.eval.stdout')).open('x') as out,(D/(arm+'.eval.stderr')).open('x') as err:
                run=subprocess.run(argv,stdout=out,stderr=err,timeout=min(30,p['deadline_epoch']-time.time()),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
            assert run.returncode==0; new=json.loads(target.read_text()); assert new['gold_sha256']==p['held_gold_sha256'] and before.get('gold_sha256',p['held_gold_sha256'])==p['held_gold_sha256'] and new['evaluator_version']==before['evaluator_version'] and new['configuration']==before['configuration']
            assert new['counts']['relation']==before['counts']['relation']
            results[arm]=dict(old_metrics=before['metrics'],metrics=new['metrics'],delta={k:new['metrics'][k]-before['metrics'][k] for k in new['metrics']},counts=new['counts'],metrics_ref=ref(target),Score_strictly_higher=new['metrics']['Score']>before['metrics']['Score'])
        save(D/'result.json',dict(status='COMPLETE_FIXED_ONE_EPOCH_TWO_ENDPOINT_CONTROL',arms=results,retain_primary=results['primary']['Score_strictly_higher'],diagnostic_does_not_replace_primary=True,quality_calls=2,old_quality_calls=0,seconds=time.monotonic()-start)); print(json.dumps(results)); return
    regression=a.mode=='regression'; out=D/('original-regression' if regression else 'new-endpoints'); out.mkdir(exist_ok=False)
    get=lambda k: p['fixed_inputs'][k]['path']
    docs=ph.keyed(ph.rows(get('input'))); bases=ph.keyed(ph.rows(get('base'))); sap=ph.keyed(ph.rows(get('sap'))); crf=ph.keyed(ph.rows(get('crf')))
    assert len(docs)==16 and set(docs)==set(bases)==set(sap)==set(crf)
    oldpool,pubmerge,biomerge,bbmerge,parse_hpo,_=ph.load_pure_sources(p['source_root'])
    _,allowed,_,_=parse_hpo(get('hpo')); allowed.discard('HP:0000118')
    oldpub={r['pmc_id']:r['candidates'] for r in ph.rows(get('old_pubmed_candidates'))}; bio={r['pmc_id']:r['candidates'] for r in ph.rows(get('bioformer_candidates'))}
    bbcanonical=ph.rows(get('biobert_canonical')); bbe={ph.fullkey(r):r for r in ph.rows(get('biobert_provenance'))}; grouped={}; score=defaultdict(set)
    for c in bbcanonical:
        k=ph.fullkey(c); assert k in bbe; score[k].add(c['score'])
        if k not in grouped: grouped[k]=dict(pmc_id=k[0],offset=k[1],length=k[2],identifier=k[3],text=c['text'])
    assert set(grouped)==set(bbe); bb=defaultdict(list)
    for k,c in grouped.items():
        e=bbe[k]; assert sorted(score[k])==e['actual_hybrid_scores']; c.update(biobert_present=True,biobert_score=float(max(score[k])),biobert_neural=e['raw_neural_match'],biobert_dictionary=e['raw_dictionary_match'],biobert_abbreviation=e['abbrev_inferred'],biobert_origin_unknown=False); bb[k[0]].append(c)
    native_spec=None
    if regression:
        pub=oldpub; canonical=ph.rows(get('old_diagnostic_canonical'))
    else:
        native=Path(a.native_dir); manifest=json.loads((native/'manifest.json').read_text()); assert manifest['training_documents']==32 and manifest['public_model_training_on_task'] is True
        trained=json.loads(Path(p['training_terminal']).read_text()); assert trained['optimizer_iterations']==20 and trained['examples']==1252
        native_spec={k:ref(native/n) for k,n in [('manifest','manifest.json'),('canonical','canonical-entities.jsonl'),('provenance','source-provenance.jsonl')]}
        pub,_=ph.public_candidates(native_spec,'pubmed',docs,p['fixed_inputs']['input']['sha256'],allowed,trained=True,expected_weight=trained['trained_H5']['sha256']); canonical=ph.rows(native/'canonical-entities.jsonl')
    records=[]
    for did,doc in docs.items():
        cs=oldpool(doc,bases[did],sap[did]['candidates'],crf[did]); cs=pubmerge(cs,pub[did]); cs=biomerge(cs,bio[did])
        records.append(bbmerge(dict(pmc_id=did,input=doc,base=bases[did],candidates=cs,origin_index=None),bb[did],doc))
    raw=ph.raw_jsonl(records); pool=out/'input.pool.jsonl.gz'; pool.write_bytes(gzip.compress(raw,mtime=0))
    if regression: assert raw==gzip.decompress(Path(get('old_pool')).read_bytes()),'Original F1 candidate reconstruction differs'
    contract=json.loads(Path(p['apply_contract_template']).read_text()); contract.update(mode='fixed-F1-regression' if regression else 'adapted-PubMed-F1-application',candidate_count=sum(len(r['candidates']) for r in records),expected_prediction_sha256=p['fixed_inputs']['primary_template']['sha256'] if regression else None)
    contract['inputs']['input_pool']=ref(pool); contract_file=out/'apply-contract.json'; save(contract_file,contract)
    argv=[p['python'],str(S/'apply_selector.py'),'--contract',str(contract_file),'--model',get('selector'),'--input-pool',str(pool),'--input',get('input'),'--relation-template',get('primary_template'),'--hpo',get('hpo'),'--dependency-root',p['dependency_root'],'--runtime-root',p['runtime_root'],'--output-dir',str(out/'primary')]
    with (out/'apply.stdout').open('x') as stdout,(out/'apply.stderr').open('x') as stderr: run=subprocess.run(argv,stdout=stdout,stderr=stderr,timeout=30,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    assert run.returncode==0,'Saved selector application failed'
    # Original pure overlay function, same negation/overlap order and old base.
    baseline=load_module('h5_original_baseline',p['sources']['baseline']['path']); ns=dict(json=json,copy=copy,Counter=__import__('collections').Counter,math=math,baseline=baseline)
    ph.functions(p['sources']['overlay']['path'],['assemble_document'],ns)
    helper=load_module('h5_literal_replace',S/'apply_selector.py'); diag_bases=ph.keyed(ph.rows(get('diagnostic_base'))); bydoc=defaultdict(list)
    for c in canonical: bydoc[c['pmc_id']].append(c)
    lines=Path(get('diagnostic_template')).read_text().splitlines(keepends=True); output=[]
    for line in lines:
        d=json.loads(line); did=d['pmc_id']; pred,_=ns['assemble_document'](docs[did],diag_bases[did],bydoc[did],allowed); output.append(helper.replace_entities(line,pred['entities']))
    diagnostic=out/'diagnostic.jsonl'; diagnostic.write_text(''.join(output))
    if regression: assert ph.sha(diagnostic)==p['fixed_inputs']['diagnostic_template']['sha256'],'Original overlay differs'
    predictions=dict(primary=ref(out/'primary/prediction.jsonl'),diagnostic=ref(diagnostic)); report=dict(mode=a.mode,documents=16,candidates=sum(len(x['candidates']) for x in records),predictions=predictions,native=native_spec,gold_read=False,fit_calls=0,seconds=time.monotonic()-start)
    save(out/'construction.json',report)
    if not regression:
        assert time.time()<p['deadline_epoch']; save(D/'paired-freeze.json',dict(predictions=predictions,native=native_spec,plan=ref(a.plan),gold_read=False,at_epoch=time.time(),all_nonentity_literal_bytes_preserved=True))
    print(json.dumps(report))
if __name__=='__main__':main()
