"""Freeze strong-union/new-only entity endpoints, then execute exactly two publicv2 calls."""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import sys
from probe import write,sha


def main():
    ap=argparse.ArgumentParser()
    for n in ('normalization','input','baseline','baseline-metrics','gold','hpo','evaluator-dir','output'):ap.add_argument('--'+n,required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write(out/'execution-start.json',{'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':sys.argv,'source_sha256':sha(__file__),'gold_opened':False,'new_v2_calls':0})
    assert sha(args.input)=='962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'
    assert sha(args.baseline)=='8cb61ba94fab8de4223acafeece83ef5a243954312d008073a42e5ba80398d6f'
    assert sha(args.baseline_metrics)=='4099a624f17f4d9c2809ae62f8da677a32d419029b1dbecefbbb37eedfa3acbd'
    ev=Path(args.evaluator_dir);assert sha(ev/'evaluate.py')=='ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781';assert sha(ev/'validate_predictions.py')=='b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f'
    sys.path.insert(0,str(ev));from validate_predictions import load_jsonl,read_hpo,validate_predictions;from evaluate import evaluate,VERSION
    assert VERSION=='patientphex-public-approx-v2'
    inputs=load_jsonl(args.input);base=load_jsonl(args.baseline);base_metrics=json.loads(Path(args.baseline_metrics).read_text());assert base_metrics['metrics']['Score']==.7286975364690524
    root=Path(args.normalization);complete=json.loads((root/'complete.json').read_text());assert complete['status']=='COMPLETE_FIXED_TASK1_NORMALIZATION';assert sha(root/'entities.json')==complete['entities_sha256'];new=json.loads((root/'entities.json').read_text())
    assert len(inputs)==len(base)==len(new)==16 and set(new)=={x['pmc_id'] for x in inputs}=={x['pmc_id'] for x in base};hpo=read_hpo(args.hpo)
    outputs={'strong-union':[],'new-only':[]};counts={};newatoms_total=0
    def atom(entity,hid):return (entity['offset'],entity['length'],hid,entity['note'])
    for doc in base:
        pmc=doc['pmc_id'];fresh=new[pmc];assert len({atom(e,e['identifier']) for e in fresh})==len(fresh)
        original_atoms={atom(e,hid) for e in doc['entities'] for hid in e['identifier'].split(';')};added=[e for e in fresh if atom(e,e['identifier']) not in original_atoms];newatoms_total+=len(added)
        union=copy.deepcopy(doc);union['entities'].extend(copy.deepcopy(added));alone=copy.deepcopy(doc);alone['entities']=copy.deepcopy(fresh)
        assert union['entities'][:len(doc['entities'])]==doc['entities']
        for output in (union,alone):assert {k:v for k,v in output.items() if k!='entities'}=={k:v for k,v in doc.items() if k!='entities'}
        outputs['strong-union'].append(union);outputs['new-only'].append(alone);counts[pmc]={'base_entities':len(doc['entities']),'new_only_atoms':len(fresh),'added_atoms':len(added)}
    refs={}
    for name,rows in outputs.items():
        report=validate_predictions(inputs,rows,hpo=hpo);write(out/(name+'.format.json'),report);assert report['valid']
        p=out/(name+'.pred.jsonl');p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));refs[name]={'prediction':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'documents':16,'patients':39,'entities':sum(len(x['entities']) for x in rows)}
    freeze={'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'endpoints':refs,'baseline_sha256':sha(args.baseline),'normalization_complete_sha256':sha(root/'complete.json'),'evaluator_sha256':sha(ev/'evaluate.py'),'validator_sha256':sha(ev/'validate_predictions.py'),'configuration':{'body_scope':'all','no_id_doc':'text','wildcard_no_id':True},'added_atoms':newatoms_total,'per_document':counts,'nonentity_objects_exact':True,'base_entity_objects_retained_exact':True,'gold_not_read_yet':True,'old_quality_calls':0,'planned_new_v2_calls':2}
    write(out/'freeze.json',freeze)
    # Gold access begins only after BOTH complete predictions are immutable and validated.
    assert sha(args.gold)=='665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3';gold=load_jsonl(args.gold);result={}
    write(out/'quality-start.json',{'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha(out/'freeze.json'),'gold_sha256':sha(args.gold),'evaluator_version':VERSION,'configuration':freeze['configuration']})
    for name,rows in outputs.items():
        assert sha(out/(name+'.pred.jsonl'))==refs[name]['sha256']
        report=evaluate(gold,rows,body_scope='all',no_id_doc='text',wildcard_no_id=True,validate=False)
        report.update(evaluator_version=VERSION,configuration=freeze['configuration'],prediction_sha256=refs[name]['sha256'],gold_sha256=sha(args.gold),freeze_sha256=sha(out/'freeze.json'),new_v2_call_index=len(result)+1)
        write(out/(name+'.metrics.json'),report);result[name]=report
        assert report['metrics']['micF1']==base_metrics['metrics']['micF1'] and report['metrics']['macF1']==base_metrics['metrics']['macF1']
    delta={name:{k:r['metrics'][k]-base_metrics['metrics'][k] for k in r['metrics']} for name,r in result.items()}
    summary={'status':'TWO_COMPLETE_TASK1_ENDPOINTS_EVALUATED','baseline_metrics':base_metrics['metrics'],'metrics':{name:r['metrics'] for name,r in result.items()},'deltas':delta,'added_atoms':newatoms_total,'marginal_mention_tp':result['strong-union']['counts']['mention']['tp']-base_metrics['counts']['mention']['tp'],'marginal_mention_fp':result['strong-union']['counts']['mention']['fp']-base_metrics['counts']['mention']['fp'],'primary_strictly_better':result['strong-union']['metrics']['Score']>base_metrics['metrics']['Score'],'new_quality_calls':2,'old_quality_calls':0,'original_dev_calls':0,'official_submissions':0,'freeze_sha256':sha(out/'freeze.json'),'endpoints':refs}
    write(out/'complete.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
