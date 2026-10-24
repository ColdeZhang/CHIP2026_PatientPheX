"""Use all fixed train32 targets and only predetermined F1 span inputs."""
import argparse,json,time
from collections import Counter,defaultdict
from pathlib import Path
from helpers import ref,write,load_jsonl,validate_refs
from reuse.strict import HPO,verify_span
from mention_schema import SYSTEM,answer,strict_parse


def messages(context,quote,start,end):
    assert context[start:end]==quote
    # Numeric original-context indices distinguish repeated identical strings without changing text.
    body={'context':context,'mention':quote,'mention_start':start,'mention_end':end}
    return [{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(body,ensure_ascii=False)}]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();start=time.monotonic();plan=json.loads(Path(a.plan).read_text());validate_refs(plan['inputs']);r=plan['inputs']
    assert plan['phase']=='CPU_PREPARATION_ONLY';h=HPO(r['hpo']['path']);names={h.terms[i]['name']:i for i in sorted(h.allowed) if i!='HP:0000118'};assert len(names)==19119
    targets=load_jsonl(r['train_targets']['path']);train=load_jsonl(r['train32']['path']);held=load_jsonl(r['F1_input']['path']);base=load_jsonl(r['strong_base']['path']);new=json.loads(Path(r['gliner_entities']['path']).read_text())
    assert len(targets)==2353 and len(train)==32 and len(held)==len(base)==len(new)==16
    train_docs={str(x['pmc_id']):x for x in train};docs={str(x['pmc_id']):x for x in held};assert not set(train_docs)&set(docs)
    assert set(docs)==set(new)=={str(x['pmc_id']) for x in base};assert sum(len(d['patient']) for d in held)==39
    assert all('entities' not in d and 'association' not in d for d in held)
    training=[]
    for row in targets:
        pmc=row['pmc_id'];assert pmc in train_docs;verify_span(train_docs[pmc],row['text'],row['offset']);text=answer(row['target'],row['text']);parsed=strict_parse(text,row['text'],names,eos=True,capped=False)
        assert parsed['status']==row['target']['status'];assert parsed['ids']==[names[x] for x in row['target']['names']] if parsed['status']=='HPO' else parsed['ids']==(['-1'] if parsed['status']=='NIL' else [])
        training.append({'row_id':row['row_id']+':mention-canonical','pmc_id':pmc,'quote':row['text'],'offset':row['offset'],'length':row['length'],'target_status':row['target']['status'],'messages':messages(row['context'],row['text'],row['mention_start_in_context'],row['mention_end_in_context']),'answer':text})
    spans=defaultdict(set)
    for d in base:
        for e in d['entities']:spans[str(d['pmc_id']),e['offset'],e['length'],e['text']].add('strong_base')
    for pmc,entities in new.items():
        for e in entities:spans[pmc,e['offset'],e['length'],e['text']].add('gliner_accepted')
    inputs=[];audit=[]
    for (pmc,offset,length,quote),origins in sorted(spans.items()):
        d=docs[pmc];pi=verify_span(d,quote,offset);assert length==len(quote);p=d['full_text'][pi];begin=offset-p['offset'];end=begin+length;left=max(0,begin-256);right=min(len(p['text']),end+256);context=p['text'][left:right]
        row={'row_id':f'{pmc}:{offset}:{length}:mention-canonical','pmc_id':pmc,'quote':quote,'offset':offset,'length':length,'messages':messages(context,quote,begin-left,end-left)};inputs.append(row);audit.append({'row_id':row['row_id'],'pmc_id':pmc,'offset':offset,'length':length,'quote':quote,'origins':sorted(origins),'passage_index':pi,'context_original_offset':p['offset']+left,'context_chars':len(context)})
    assert len({x['row_id'] for x in training+inputs})==len(training)+len(inputs)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);arts={'train':write(out/'train.jsonl',training,jsonl=True),'held_input':write(out/'F1.input.jsonl',inputs,jsonl=True),'held_span_audit':write(out/'F1.span-audit.json',audit),'public_names':write(out/'public-names.json',names)}
    counts=dict(Counter(x['target_status'] for x in training));assert counts=={'HPO':2222,'NIL':44,'NO':87}
    manifest={'status':'COMPLETE_FIXED_MENTION_TARGET_AND_INPUT_ONLY_PREPARATION','source':ref(__file__),'plan':ref(a.plan),'inputs':r,'artifacts':arts,'train_documents':32,'train_rows':2353,'train_status':counts,'held_documents':16,'held_patients':39,'held_rows':len(inputs),'held_origin_combinations':dict(Counter('+'.join(x['origins']) for x in audit)),'base_entities':sum(len(x['entities']) for x in base),'gliner_entities':sum(len(x) for x in new.values()),'held_labels_or_predicted_ids_not_prompted':True,'no_gold_model_GPU_fit_quality':True,'seconds':time.monotonic()-start}
    write(out/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ['source','plan','inputs','artifacts']}))

if __name__=='__main__':main()
