"""T5-specific complete raw verification, then the unchanged original G itemwise parser."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from common import ref,save,fsync_dir
from runtime import same
from transport import decode_generated


def main(a):
    control=json.loads(Path(a.control).read_text());plan=json.loads(Path(a.formal_plan).read_text())
    forbidden=os.path.realpath(control['gold_metadata_only']['path'])
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0]))==forbidden:
            raise PermissionError('Gold forbidden during T5 raw extraction')
    sys.addaudithook(audit)
    for e in control['references'].values():same(e['path'],e)
    sys.path.insert(0,control['grounded_worktree'])
    from solutions.grounded_generation.inference.decode import interpret
    import sentencepiece as spm
    same(a.spiece,plan['model']['spiece.model']);sp=spm.SentencePieceProcessor(model_file=a.spiece)
    same(a.supply,plan['supply']);supply=json.loads(Path(a.supply).read_text());held=Path(a.supply).parent/Path(supply['artifacts']['held_input']['path']).name
    same(held,supply['artifacts']['held_input']);inputs=[json.loads(x) for x in gzip.decompress(held.read_bytes()).splitlines()]
    docs=[json.loads(x) for x in Path(control['references']['F1_input']['path']).read_text().splitlines()];by={str(d['pmc_id']):d for d in docs}
    assert len(inputs)==39 and len({r['row_id'] for r in inputs})==39 and len(by)==16
    assert all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs)
    expected={(str(d['pmc_id']),p['patient_id']) for d in docs for p in d['patient']}
    assert len(expected)==39 and {(str(r['pmc_id']),r['patient_id']) for r in inputs}==expected
    terminal=json.loads(Path(a.terminal).read_text());assert terminal['exit_code']==0 and terminal['phase_complete'] and not terminal['phase_failure']
    complete=json.loads(Path(a.complete).read_text());assert complete['status']=='COMPLETE_LONGT539_FINAL_EPOCH10_GREEDY1600_RAW'
    assert complete['rows']==39 and complete['documents']==16 and complete['row_ids']==[r['row_id'] for r in inputs]
    same(a.formal_plan,complete['plan']);same(Path(__file__).parent/'infer.py',complete['source'])
    assert complete['source']['sha256']==plan['sources']['formal/infer.py']['sha256']
    assert complete['supply']['sha256']==plan['supply']['sha256'] and complete['no_regeneration_or_answer_repair']
    remote=Path(complete['responses']['path']).parent;local=Path(a.complete).resolve().parent
    def mapped(e):
        relative=Path(e['path']).relative_to(remote);assert '..' not in relative.parts
        path=local/relative;same(path,e);return path
    records=[json.loads(x) for x in mapped(complete['responses']).read_text().splitlines()]
    assert len(records)==len(complete['generation_refs'])==len(complete['raw_refs'])==39
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);decisions=[];quotes=[]
    for index,(row,record,gref,rref) in enumerate(zip(inputs,records,complete['generation_refs'],complete['raw_refs'])):
        gen=json.loads(mapped(gref).read_text());raw=json.loads(mapped(rref).read_text());assert raw==record
        assert all(raw[k]==v for k,v in gen.items()) and raw['generation']==gref
        assert gen['index']==index and gen['row_id']==row['row_id'] and str(gen['pmc_id'])==str(row['pmc_id'])
        assert gen['target_patient_id']==row['patient_id'] and gen['source_sha256']==row['source_sha256']
        assert gen['input_token_count']==len(row['input_ids']) and sp.encode(row['source_text'],out_type=int)+[1]==row['input_ids']
        ids=gen['generated_token_ids'];assert gen['all_generated_ids']==[0]+ids and len(ids)==gen['output_tokens']<=1600
        assert all(type(x) is int and 0<=x<32128 for x in ids)
        assert gen['actual_last_token_eos']==bool(ids and ids[-1]==1) and gen['hit_token_limit']==(len(ids)>=1600)
        assert 1 not in ids[:-1] and gen['decoded_or_parsed'] is False and raw['raw_before_parse'] and raw['parsed'] is False
        response=None;error=None
        try:response=decode_generated(sp,ids)
        except ValueError as exc:error=str(exc)
        assert response==raw['response'] and error==raw['transport_error']
        assert raw['native_sp_diagnostic']==sp.decode([x for x in ids if x<sp.get_piece_size()])
        doc=by[str(row['pmc_id'])]
        decision=interpret(response if error is None else '',doc,gen['actual_last_token_eos'],gen['hit_token_limit'])
        if error is not None and gen['actual_last_token_eos'] and not gen['hit_token_limit']:
            assert not decision['anchors'];decision['reason']='undefined_generated_transport_ID';decision['transport_error']=error
        decisions.append({'row_id':row['row_id'],'pmc_id':str(row['pmc_id']),'target_patient_id':row['patient_id'],'raw':ref(mapped(rref)),'decision':decision})
        for j,anchor in enumerate(decision['anchors']):quotes.append({'row_id':row['row_id'],'pmc_id':str(row['pmc_id']),'target_patient_id':row['patient_id'],'anchor_index':j,**anchor})
    with (out/'quotes.jsonl').open('x') as f:
        for q in quotes:f.write(json.dumps(q,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
    save(out/'decisions.json',decisions)
    save(out/'manifest.json',{'schema':'longt5-extraction-complete-v1','source':ref(__file__),'control':ref(a.control),'formal_plan':ref(a.formal_plan),
        'complete':ref(a.complete),'terminal':ref(a.terminal),'supply':ref(a.supply),'original_G_decode':control['references']['decode'],
        'generated_transport':plan['sources']['formal/transport.py'],'complete_original39':True,'documents':16,'patients':39,
        'nonEOS_cap_JSON_whole_row_abstention':True,'itemwise_exact_grounding':True,'exact_anchors':len(quotes),
        'unique_quotes':len({q['quote'] for q in quotes}),'quotes':ref(out/'quotes.jsonl'),'decisions':ref(out/'decisions.json'),
        'abstaining_rows':sum(x['decision']['status']=='abstain' for x in decisions),'rejected_items':sum(len(x['decision']['rejected']) for x in decisions),
        'row_reason_counts':dict(Counter(x['decision']['reason'] for x in decisions)),'original_raw_not_repaired':True,
        'HPO_mapping_or_gold_or_score_performed':False,'HF_decode_equality_claimed':False})
    fsync_dir(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('control','formal-plan','complete','terminal','supply','spiece','output'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
