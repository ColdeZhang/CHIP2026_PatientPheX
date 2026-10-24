"""Complete80 local redecoding; original79 exact, common freeze, at most one v2."""
import argparse
import importlib.util
import json
from pathlib import Path
import signal
import sys
from codec import decode_set
from contracts import run as contracts
sys.dont_write_bytecode=True

def load(path):
    p=json.loads(Path(path).read_text());r=p['refs'];spec=importlib.util.spec_from_file_location('_set_io',r['helper']['path']);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H);H.checked(r['helper']);gate=H.gold_guard(p)
    assert p['source']==H.ref(__file__) and p['policy']=='Stable deduplication of identical legal PT values within each CID only'
    assert p['baseline_score']==.6866189430196797
    for item in r.values():H.checked(item)
    for item in p['sources'].values():H.checked(item)
    original=H.load_module('_original_response_codec',r['original_codec'])
    sys.path.insert(0,str(Path(r['render_helper']['path']).parent));H.checked(r['literal_helper'])
    renderer=H.load_module('_original_coded_renderer',r['render_helper'])
    return p,H,gate,original,renderer

def assemble(path):
    p,H,gate,original,renderer=load(path);r=p['refs'];out=Path(p['output']);out.mkdir(parents=True,exist_ok=False)
    H.save(out/'contracts.json',contracts(original));manifest=H.read(r['terminal_manifest']['path']);terminal=Path(p['terminal'])
    for item in manifest['files']:H.checked(dict(path=str(terminal/item['name']),bytes=item['bytes'],sha256=item['sha256']))
    prompts=H.jsonl(r['prompts']['path']);responses=H.jsonl(terminal/'responses.jsonl');oldlines=(terminal/'canonical.jsonl').read_bytes().decode().splitlines(keepends=True);old=[json.loads(x) for x in oldlines]
    assert len(prompts)==len(responses)==len(old)==80 and all('answer' not in row for row in prompts)
    pm={x['row_id']:(i,x) for i,x in enumerate(prompts)};rm={x['row_id']:x for x in responses};assert len(pm)==len(rm)==80 and set(pm)==set(rm)=={x['row_id'] for x in old}
    base=H.jsonl(r['baseline']['path']);decisions={str(d['pmc_id']):{str(a['patient_id']):set() for a in d['association']} for d in base};prior={doc:{pid:set() for pid in m} for doc,m in decisions.items()}
    assert (len(base),sum(len(d['entities']) for d in base),sum(len(d['association']) for d in base))==(16,1189,39)
    newlines=[];audit=[];unchanged=changed=valid=0;candidates=pairs=0;infer_plan=H.read(r['inference_plan']['path'])
    for oldline,c in zip(oldlines,old):
        i,row=pm[c['row_id']];raw=rm[c['row_id']];ids=H.read(terminal/f'generated-ids/{i:03d}.json')
        assert H.read(terminal/f'raw-responses/{i:03d}.json')==raw and H.read(terminal/f'canonical-records/{i:03d}.json')==c
        assert str(c['pmc_id'])==str(row['pmc_id'])==str(raw['pmc_id']) and ids['row_id']==row['row_id']
        assert len(ids['generated_ids'])==raw['output_tokens'] and ids['configured_eos_ids']==infer_plan['configured_eos_ids']
        natural=bool(ids['generated_ids'] and ids['generated_ids'][-1] in ids['configured_eos_ids']);cap=len(ids['generated_ids'])>=infer_plan['generation_limit'] and not natural
        assert natural==raw['natural_eos']==ids['natural_eos'] and cap==raw['hit_token_limit']==ids['hit_token_limit']
        decoded=None;folds=[];error=None
        try:decoded,folds=decode_set(raw['response'],row,original,natural_eos=natural,hit_token_limit=cap)
        except (ValueError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
        if c['strict_complete']:
            assert decoded==c['candidate_patients'] and folds==[] and error is None and c['abstention'] is None
            newline=oldline;unchanged+=1
        else:
            assert c['candidate_patients']=={} and c['abstention'];changed+=int(decoded is not None)
            updated={**c,'candidate_patients':decoded if decoded is not None else {},'strict_complete':decoded is not None,'abstention':error}
            newline=json.dumps(updated,ensure_ascii=False)+'\n'
        newlines.append(newline);valid+=int(decoded is not None);doc=str(row['pmc_id'])
        for matrix,target in [(c['candidate_patients'],prior),(decoded or {},decisions)]:
            if matrix:
                assert set(matrix)==set(row['code_map'])
                for cid,pids in matrix.items():
                    assert len(pids)==len(set(pids)) and set(pids)<=set(row['patient_ids'])
                    for pid in pids:target[doc][pid].add(row['code_map'][cid])
        audit.append(dict(row_id=row['row_id'],original_index=i,raw=H.ref(terminal/f'raw-responses/{i:03d}.json'),ids=H.ref(terminal/f'generated-ids/{i:03d}.json'),
            original_strict_complete=c['strict_complete'],new_set_complete=decoded is not None,original_record=c,new_record=json.loads(newline),duplicate_values_folded=folds,raw_text_and_IDs_unchanged=True))
        candidates+=len(row['code_map']);pairs+=len(row['code_map'])*len(row['patient_ids'])
    assert unchanged==79 and (candidates,pairs)==(587,1557)
    H.publish(out/'canonical-set.jsonl',''.join(newlines).encode());H.publish(out/'decode-audit.jsonl',''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in audit).encode())
    original_pred=renderer.render(r['render_template']['path'],prior,out/'original-reconstructed.jsonl')
    assert Path(original_pred['path']).read_bytes()==Path(r['baseline']['path']).read_bytes()
    predicted=renderer.render(r['render_template']['path'],decisions,out/'prediction.jsonl')
    validator=H.load_module('validate_predictions',r['validator']);v=validator.validate_predictions(validator.load_jsonl(r['input']['path']),validator.load_jsonl(predicted['path']),validator.read_hpo(r['hpo']['path']))
    H.save(out/'format.json',v);assert v['valid'] and not gate['enabled']
    same=Path(predicted['path']).read_bytes()==Path(r['baseline']['path']).read_bytes()
    H.save(out/'freeze.json',dict(plan=H.ref(path),source=H.ref(__file__),prediction=predicted,original_reconstructed=original_pred,canonical=H.ref(out/'canonical-set.jsonl'),
        audit=H.ref(out/'decode-audit.jsonl'),contracts=H.ref(out/'contracts.json'),format=H.ref(out/'format.json'),rows=80,documents=16,patients=39,entities=1189,
        original79_lines_byteexact=True,new_valid_rows=valid,original_failed_now_valid=changed,duplicate_occurrences_folded=sum(len(x['duplicate_values_folded']) for x in audit),
        baseline_relations=sum(len(x) for m in prior.values() for x in m.values()),new_relations=sum(len(x) for m in decisions.values() for x in m.values()),
        byteexact_baseline=same,gold_read=False,new_v2=0,old_v2=0,NN_calls=0,model_tokenizer_imports=0,original_generator_provenance_fresh_not_new_generation=True))
    print(json.dumps(H.ref(out/'freeze.json')))

def score(path):
    p,H,gate,_,_=load(path);r=p['refs'];out=Path(p['output']);f=H.read(out/'freeze.json');assert f['plan']==H.ref(path) and f['source']==H.ref(__file__)
    for key in ['prediction','original_reconstructed','canonical','audit','contracts','format']:H.checked(f[key])
    baseline=H.read(r['baseline_metrics']['path']);old=baseline['metrics'];assert old['Score']==p['baseline_score']
    ev=out/'evaluation';ev.mkdir(exist_ok=False)
    if f['byteexact_baseline']:
        assert Path(f['prediction']['path']).read_bytes()==Path(r['baseline']['path']).read_bytes() and not gate['enabled']
        H.save(ev/'result.json',dict(status='BYTEEXACT_ZERO_GOLD_ZERO_V2',baseline=old,keep=False,new_v2=0,old_v2=0,NN_calls=0,freeze=H.ref(out/'freeze.json')));return
    H.save(ev/'before-gold.json',dict(freeze=H.ref(out/'freeze.json'),gold_not_yet_opened=True,new_v2=1,old_v2=0))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Unique CPU score30s')));signal.alarm(30)
    try:
        validator=H.load_module('validate_predictions',r['validator']);evaluator=H.load_module('_response_set_v2',r['evaluator']);assert evaluator.VERSION=='patientphex-public-approx-v2' and evaluator.validate_predictions is validator.validate_predictions
        gate['enabled']=True;H.checked(p['gold']);gold=validator.load_jsonl(p['gold']['path']);pred=validator.load_jsonl(f['prediction']['path'])
        H.save(ev/'call-start.json',dict(freeze=H.ref(out/'freeze.json'),new_v2=1,old_v2=0,body_scope='all',no_id_doc='text'))
        measured=evaluator.evaluate_report(gold,pred,'all','text');H.save(ev/'metrics.json',measured);delta={k:measured['metrics'][k]-old[k] for k in H.METRICS};assert delta['menF1']==delta['docF1']==0
        before=baseline['counts']['relation'];after=measured['counts']['relation']
        H.save(ev/'result.json',dict(status='COMPLETE_UNIQUE_RESPONSE_SET_F1',metrics=measured['metrics'],baseline=old,delta=delta,keep=delta['Score']>0,
            relation_delta={k:after[k]-before[k] for k in ['tp','fp','fn']},new_v2=1,old_v2=0,NN_calls=0,union_or_scan_calls=0,
            development_after_observed_failure=True,blind_test=False,F0_read=False,original_dev_read=False,freeze=H.ref(out/'freeze.json'),metrics_ref=H.ref(ev/'metrics.json')))
        print(json.dumps(H.ref(ev/'result.json')))
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assemble','score']);p.add_argument('--plan',required=True);a=p.parse_args();globals()[a.stage](a.plan)
