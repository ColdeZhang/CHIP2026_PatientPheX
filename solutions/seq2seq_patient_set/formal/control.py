"""One prelocked F1 support endpoint; no pure-T5 quality or threshold/seed sweep."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from common import ref,save,fsync_dir
from runtime import same


def pairs(rows, evaluator):
    return {(str(d['pmc_id']),pid,h) for d in rows for pid,hs in evaluator.associations(d).items() for h in hs if h!='NO'}


def primary(base, t5, support):
    return base | (t5 & support)


def main(a):
    c=json.loads(Path(a.control).read_text());assert c['schema']=='longt5-fixed-support-control-v1'
    guard={'gold':False};forbidden=os.path.realpath(c['gold_metadata_only']['path'])
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0]))==forbidden and not guard['gold']:
            raise PermissionError('Gold forbidden before the sole complete T5 freeze')
    sys.addaudithook(audit)
    for e in c['references'].values():same(e['path'],e)
    same(c['support_ref']['path'],c['support_ref']);support_refs=json.loads(Path(c['support_ref']['path']).read_text())['F1']
    sys.path.insert(0,c['quality_source_root']);import support,assemble
    from verify_normalization import verify_normalization
    sys.path.insert(0,str(Path(c['references']['evaluator']['path']).parent));import evaluate as ev,validate_predictions as va
    out=Path(a.output)
    if a.stage=='assemble':
        assert a.actual
        actual=json.loads(Path(a.actual).read_text())
        for e in actual.values():same(e['path'],e)
        ex=json.loads(Path(actual['extraction_manifest']['path']).read_text())
        assert ex['schema']=='longt5-extraction-complete-v1' and ex['control']==ref(a.control)
        same(ex['source']['path'],ex['source']);assert ex['source']==ref(Path(__file__).parent/'extract.py')
        assert ex['complete_original39'] and ex['documents']==16 and ex['patients']==39 and ex['original_raw_not_repaired']
        assert ex['quotes']==actual['quotes'] and ex['decisions']==actual['decisions'] and not ex['HPO_mapping_or_gold_or_score_performed']
        for k in ('complete','terminal','supply','formal_plan'):same(ex[k]['path'],ex[k])
        complete=json.loads(Path(ex['complete']['path']).read_text());terminal=json.loads(Path(ex['terminal']['path']).read_text())
        assert complete['rows']==39 and terminal['exit_code']==0 and terminal['phase_complete']
        decisions=json.loads(Path(actual['decisions']['path']).read_text());assert len(decisions)==39
        quotes=support.jsonl(actual['quotes']['path']);rebuilt=[]
        for row in decisions:
            if row['decision']['status']=='abstain':assert not row['decision']['anchors']
            for index,anchor in enumerate(row['decision']['anchors']):rebuilt.append({'row_id':row['row_id'],'pmc_id':row['pmc_id'],'target_patient_id':row['target_patient_id'],'anchor_index':index,**anchor})
        assert quotes==rebuilt and len(quotes)==ex['exact_anchors']
        p={'actual':actual,'references':c['references']};normalized=verify_normalization(p,ex)
        inputs=support.jsonl(c['references']['F1_input']['path']);strict=support.load_module('_longt5_strict',c['references']['strict'])
        hpo=strict.HPO(c['references']['hpo']['path']);schema=support.load_module('_longt5_schema',c['references']['schema'])
        class BoundStrict:
            HPO=hpo
            verify_span=staticmethod(strict.verify_span)
        selected,_,counts=assemble.new_relations(normalized,inputs,BoundStrict,schema,c['references']['normalization_policy']['sha256'])
        baseline=c['F1_baseline']['base'];same(baseline['path'],baseline)
        lines,base,ids=support.rows(baseline['path']);old=pairs(base,ev)
        for key in ('G','S','Q','C','inherited_baseline'):same(support_refs[key]['path'],support_refs[key])
        views={key:pairs(support.jsonl(support_refs[key]['path']),ev) for key in ('G','S','Q','C')};union=set().union(*views.values())
        inherited=pairs(support.jsonl(support_refs['inherited_baseline']['path']),ev);assert inherited<=old and inherited<=views['S']
        assert len(old)==524 and len(union-old)==133 and len(base)==16 and sum(len(d['association']) for d in base)==39
        t5={(did,pid,key[1]) for (did,pid),keys in selected.items() for key in keys};final=primary(old,t5,union);added=final-old
        spans,mask=support.byte_helpers(c['references']['byte_helper']);rendered=[]
        for line,d in zip(lines,base):
            values={r['patient_id']:r['phenotype']+sorted(h for did,pid,h in added if did==str(d['pmc_id']) and pid==r['patient_id']) for r in d['association']}
            rendered.append(support.replace_phenotypes(line,values,spans,mask))
        raw=''.join(rendered).encode();pred=[json.loads(x) for x in rendered]
        assert pairs(pred,ev)==final and all(mask(x)==mask(y) for x,y in zip(lines,rendered))
        out.mkdir(parents=True,exist_ok=False)
        with (out/'prediction.jsonl').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        fmt=va.validate_predictions(inputs,pred,va.read_hpo(c['references']['hpo']['path']));save(out/'format.json',fmt);assert fmt['valid']
        save(out/'freeze.json',{'source':ref(__file__),'control':ref(a.control),'actual':ref(a.actual),'prediction':ref(out/'prediction.jsonl'),
            'formula':c['formula'],'baseline_relations':524,'support_outside_baseline':133,'T5_relations':len(t5),'added':len(added),
            'added_pairs':sorted(added),'removed':0,'nonphenotype_bytes_exact':True,'baseline_prefixes_exact':True,'no_gold_opened':True,
            'new_v2':0,'old_v2':0,'normalization_counts':counts,'byteexact_to_baseline':raw==Path(baseline['path']).read_bytes(),'at_unix':time.time()})
        fsync_dir(out)
    else:
        frozen=json.loads((out/'freeze.json').read_text());assert frozen['source']==ref(__file__) and frozen['control']==ref(a.control)
        same(frozen['prediction']['path'],frozen['prediction']);same(frozen['actual']['path'],frozen['actual'])
        old_ref=c['F1_baseline']['old_metrics'];same(old_ref['path'],old_ref);old=json.loads(Path(old_ref['path']).read_text())['metrics']
        assert old['Score']==0.701325377212172
        intent=out/'evaluation-intent.json'
        with intent.open('x') as f:
            json.dump({'freeze':ref(out/'freeze.json'),'new_v2':0 if frozen['byteexact_to_baseline'] else 1,'old_v2':0,'unix':time.time()},f);f.flush();os.fsync(f.fileno())
        fsync_dir(out)
        if frozen['byteexact_to_baseline']:
            save(out/'result.json',{'status':'CLOSED_BYTEEXACT_NO_INCREMENT','metrics':old,'new_v2':0,'old_v2':0,'strict_positive':False});return
        guard['gold']=True;same(c['gold_metadata_only']['path'],c['gold_metadata_only'])
        gold=va.load_jsonl(c['gold_metadata_only']['path']);pred=va.load_jsonl(frozen['prediction']['path']);report=ev.evaluate_report(gold,pred,body_scope='all',no_id_doc='text')
        save(out/'metrics.json',report);delta={k:report['metrics'][k]-v for k,v in old.items()};assert delta['menF1']==delta['docF1']==0
        additions=set(map(tuple,frozen['added_pairs']));tp=len(additions & pairs(gold,ev))
        save(out/'result.json',{'status':'COMPLETE_SINGLE_LONGT5_FIXED_F1_SUPPORT_CONTROL','metrics':report['metrics'],'baseline':old,'delta':delta,
            'added':len(additions),'added_TP':tp,'added_FP':len(additions)-tp,'strict_positive':delta['Score']>0,'new_v2':1,'old_v2':0,
            'not_blind':True,'NN_calls':0,'fit_calls':0,'freeze':ref(out/'freeze.json'),'F0_allowed':delta['Score']>0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assemble','score']);p.add_argument('--control',required=True)
    p.add_argument('--actual');p.add_argument('--output',required=True);main(p.parse_args())
