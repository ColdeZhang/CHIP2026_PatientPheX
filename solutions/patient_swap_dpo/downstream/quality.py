"""One fixed DPO-replaced four-source F1 endpoint; no model or parameter search."""
import argparse
from collections import Counter
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from common import checked,ref,save,sha,stamp
from codec import parse_complete
from literal_helpers import association_span,nonrelation_bytes

FORMULA='B471 UNION at_least_two(G,S,Q_DPO,C)'
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x]
def atoms(value):return value.split(';') if re.fullmatch(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+',value) else [value]
def pairs(docs):return {(str(d['pmc_id']),a['patient_id'],v) for d in docs for a in d['association'] for x in a['phenotype'] for v in atoms(x)}
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def same_content(local,remote):
    actual=ref(local)
    assert all(actual[k]==remote[k] for k in ['bytes','sha256'])
def source_pairs(docs,patients):
    assert len(docs)==len(patients) and {str(d['pmc_id']) for d in docs}==set(patients)
    for d in docs:
        ids=[a['patient_id'] for a in d['association']]
        assert len(ids)==len(set(ids)) and set(ids)==patients[str(d['pmc_id'])]
    return pairs(docs)
def sampled(diagnostics,patients):
    streams={'S1':set(),'S2':set()};seen={'S1':{},'S2':{}}
    for r in diagnostics:
        stream=r['stream']
        if stream=='G':continue
        assert stream in streams and r['row_id'] not in seen[stream]
        assert r['strict_complete'] and r['actual_natural_eos'] and r['fresh']
        assert not any(r[k] for k in ['actual_hit_token_limit','actual_has_think_marker','whole_row_abstention','parser_error'])
        doc=str(r['pmc_id']);assert doc in patients
        for d,p,value in r['relations']:
            assert str(d)==doc and p in patients[doc]
            streams[stream].update((doc,p,a) for a in atoms(value))
        seen[stream][r['row_id']]=doc
    assert seen['S1']==seen['S2'] and len(seen['S1'])==80 and set(seen['S1'].values())==set(patients)
    return streams['S1']&streams['S2']
def select(protected,sources):
    assert set(sources)==set('GSQC')
    counts=Counter(k for source in sources.values() for k in source)
    return protected|{k for k,n in counts.items() if n>=2}
def assemble(lines,desired):
    base=[json.loads(x) for x in lines];protected=pairs(base);assert protected<=desired
    output=[]
    for line,old in zip(lines,base):
        new=copy.deepcopy(old['association'])
        for a in new:
            a['phenotype'].extend(sorted(v for d,p,v in desired-protected if d==str(old['pmc_id']) and p==a['patient_id']))
        begin,end=association_span(line)
        output.append(line[:begin]+json.dumps(new,ensure_ascii=False,separators=(',',':'))+line[end:])
    docs=[json.loads(x) for x in output];assert pairs(docs)==desired
    assert nonrelation_bytes(output)==nonrelation_bytes(lines)
    for old,new in zip(base,docs):
        assert {k:v for k,v in old.items() if k!='association'}=={k:v for k,v in new.items() if k!='association'}
        for a,b in zip(old['association'],new['association']):
            assert {k:v for k,v in a.items() if k!='phenotype'}=={k:v for k,v in b.items() if k!='phenotype'}
            assert b['phenotype'][:len(a['phenotype'])]==a['phenotype']
    return output
def current_q(prompts,encoded,raw,canonical,native,patients):
    assert len(prompts)==len(encoded)==len(raw)==len(canonical)==80
    assert len({p['row_id'] for p in prompts})==80
    result=set();seen=set();valid=0;artifacts=[]
    for i,(p,e,r,c) in enumerate(zip(prompts,encoded,raw,canonical)):
        assert p['row_id']==r['row_id']==c['row_id']
        assert e['row_id']==p['row_id'] and e['code_map']==p['code_map'] and e['patient_ids']==p['patient_ids']
        assert r['input_tokens']==len(e['input_ids'])
        assert str(p['pmc_id'])==str(r['pmc_id'])==str(c['pmc_id'])
        doc=str(p['pmc_id']);assert set(p['patient_ids'])==patients[doc]
        assert r['original_index']==c['original_index']==i
        assert r['provenance']==c['provenance']=='fresh'
        path=native/'generated-ids'/f'{i:03d}.json';ids=json.loads(path.read_text());artifacts.append(ref(path))
        assert ids['row_id']==p['row_id'] and str(ids['pmc_id'])==doc and ids['original_index']==i
        tokens=ids['generated_ids'];assert all(type(x) is int for x in tokens)
        assert len(tokens)==r['output_tokens'] and len(tokens)<=1600
        natural=bool(tokens and tokens[-1] in ids['configured_eos_ids']);cap=len(tokens)>=1600 and not natural
        assert ids['natural_eos']==r['natural_eos']==c['natural_eos']==natural
        assert ids['hit_token_limit']==r['hit_token_limit']==c['hit_token_limit']==cap
        assert list(p['code_map'])==[f'C{j:03d}' for j in range(len(p['code_map']))]
        for concept in p['code_map'].values():assert (doc,concept) not in seen;seen.add((doc,concept))
        value,error=None,None
        try:
            if not natural or cap:raise ValueError('No natural EOS or generation reached cap')
            value=parse_complete(r['response'],p)
        except (ValueError,TypeError) as exc:error=type(exc).__name__+': '+str(exc)
        assert c['strict_complete']==(value is not None)
        assert c['candidate_patients']==(value if value is not None else {}) and c['abstention']==error
        for subdir,record in [('raw-responses',r),('canonical-records',c)]:
            item=native/subdir/f'{i:03d}.json';assert json.loads(item.read_text())==record;artifacts.append(ref(item))
        if value is None:continue
        valid+=1
        for cid,pids in value.items():
            for pid in pids:result.update((doc,pid,a) for a in atoms(p['code_map'][cid]))
    assert len(seen)==587
    return result,valid,artifacts
def main(plan_path,mode):
    p=json.loads(Path(plan_path).read_text());assert p['enabled'] is True and p['formula']==FORMULA
    r=p['refs'];out=Path(p['output']);allow_gold=[False]
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and '.gold.jsonl' in os.fsdecode(args[0]):
            assert allow_gold[0] and os.path.realpath(os.fsdecode(args[0]))==os.path.realpath(r['gold']['path'])
    sys.addaudithook(guard)
    for name,digest in p['source_files'].items():assert sha(Path(__file__).parent/name)==digest
    for name,item in r.items():
        if name!='gold':checked(item)
    va=module(r['validator']['path'],'validate_predictions');ev=module(r['evaluator']['path'],'dpo_frozen_evaluator')
    assert ev.VERSION=='patientphex-public-approx-v2'
    old=json.loads(Path(r['baseline_metrics']['path']).read_text())['metrics'];assert old['Score']==.701325377212172
    if mode=='assemble':
        out.mkdir(exist_ok=False,parents=True);save(out/'started.json',{'at_utc':stamp(),'plan':ref(plan_path)})
        for item in p['completed_bindings'].values():checked(item)
        formal=json.loads(Path(p['completed_bindings']['formal_complete']['path']).read_text());ack=json.loads(Path(p['completed_bindings']['persistent_ack']['path']).read_text())
        assert formal['global_step']==10 and formal['completed_epochs']==1 and formal['acknowledgement']==ack
        assert formal['counts']=={'pairs':74,'sequences':148,'tokens':519954,'supervised':9344}
        driver=json.loads(Path(p['completed_bindings']['driver_result']['path']).read_text());complete=json.loads(Path(p['completed_bindings']['native_complete']['path']).read_text())
        assert driver['status']=='COMPLETE_FIXED_DPO_F1_ALL80' and driver['return_code']==0 and not driver['timed_out'] and not driver['release']['target_apps']
        assert complete['status']=='COMPLETE_FIXED_DPO_F1_ALL80_FRESH' and complete['rows']==80 and complete['cached']==0
        native=Path(p['collected_native']);same_content(native/'responses.jsonl',complete['responses']);same_content(native/'canonical.jsonl',complete['canonical'])
        lines=Path(r['protected_base']['path']).read_text().splitlines(keepends=True);base=[json.loads(x) for x in lines]
        baseline=rows(r['baseline']['path']);inputs=rows(r['input']['path']);prompts=rows(r['prompts']['path'])
        assert len(base)==len(baseline)==len(inputs)==16 and sum(len(x['entities']) for x in base)==1189
        assert [str(x['pmc_id']) for x in base]==[str(x['pmc_id']) for x in baseline]==[str(x['pmc_id']) for x in inputs]
        patients={str(x['pmc_id']):{y['patient_id'] for y in x['patient']} for x in inputs};assert sum(map(len,patients.values()))==39
        protected=source_pairs(base,patients);prior=source_pairs(baseline,patients);assert len(protected)==471 and len(prior)==524
        sources={k:source_pairs(rows(checked(v)),patients) for k,v in p['pure_sources'].items()}
        sources['S']=sampled(rows(checked(p['sampled_diagnostics'])),patients)
        assert select(protected,sources)==prior
        encoded=[json.loads(x) for x in gzip.decompress(Path(r['held_encoded']['path']).read_bytes()).splitlines()]
        assert sum(len(x['input_ids']) for x in encoded)==666591 and max(len(x['input_ids']) for x in encoded)==14322
        new_q,strict,rawfiles=current_q(prompts,encoded,rows(native/'responses.jsonl'),rows(native/'canonical.jsonl'),native,patients)
        assert strict==complete['strict_complete'];sources['Q']=new_q;desired=select(protected,sources)
        prediction=assemble(lines,desired)
        if desired==prior:prediction=Path(r['baseline']['path']).read_text().splitlines(keepends=True)
        pred=[json.loads(x) for x in prediction];assert pairs(pred)==desired
        val=va.validate_predictions(inputs,pred,va.read_hpo(r['hpo']['path']));save(out/'validation.json',val);assert val['valid'] and val['warning_count']==0
        with (out/'prediction.jsonl').open('x') as f:f.writelines(prediction);f.flush();os.fsync(f.fileno())
        save(out/'raw-files.json',rawfiles)
        save(out/'freeze.json',{'at_utc':stamp(),'plan':ref(plan_path),'prediction':ref(out/'prediction.jsonl'),'raw_files':ref(out/'raw-files.json'),'validation':ref(out/'validation.json'),'strict_rows':strict,'whole_row_abstentions':80-strict,'source_counts':{k:len(v) for k,v in sources.items()},'protected':471,'baseline':524,'final':len(desired),'added':sorted(desired-prior),'removed':sorted(prior-desired),'same_semantic_set':desired==prior,'inherited_exact_B4_if_unchanged':desired==prior,'gold_reads':0,'quality_calls':0})
    else:
        freeze=json.loads((out/'freeze.json').read_text());assert freeze['plan']==ref(plan_path)
        for key in ['prediction','raw_files','validation']:checked(freeze[key])
        if freeze['same_semantic_set']:
            assert Path(freeze['prediction']['path']).read_bytes()==Path(r['baseline']['path']).read_bytes()
            save(out/'result.json',{'status':'ZERO_SEMANTIC_CHANGE_INHERITED_EXACT_B4','new_v2':0,'old_v2':0,'metrics':old,'retain':False,'freeze':ref(out/'freeze.json')});return
        save(out/'evaluation-intent.json',{'at_utc':stamp(),'freeze':ref(out/'freeze.json'),'new_v2':1,'old_v2':0});allow_gold[0]=True
        gold=rows(checked(r['gold']));pred=rows(freeze['prediction']['path']);metrics=ev.evaluate_report(gold,pred,'all','text');save(out/'metrics.json',metrics)
        delta={k:v-old[k] for k,v in metrics['metrics'].items()};assert delta['menF1']==delta['docF1']==0
        truth=pairs(gold);added=set(map(tuple,freeze['added']));removed=set(map(tuple,freeze['removed']))
        save(out/'result.json',{'status':'COMPLETE_FIXED_DPO_Q_REPLACEMENT_F1','at_utc':stamp(),'metrics':metrics['metrics'],'baseline':old,'delta':delta,'added':len(added),'added_TP':len(added&truth),'added_FP':len(added-truth),'removed':len(removed),'removed_TP':len(removed&truth),'removed_FP':len(removed-truth),'retain':delta['Score']>0,'new_v2':1,'old_v2':0,'development_not_blind':True,'freeze':ref(out/'freeze.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('mode',choices=['assemble','evaluate']);a=p.parse_args();main(a.plan,a.mode)
