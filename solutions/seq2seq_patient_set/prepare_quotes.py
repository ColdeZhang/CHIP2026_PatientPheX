"""Freeze original G87 quote supervision, complete39 inputs; no drop or new weak targets."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from census import ref,rows
import quote_codec


def main(a):
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=a.spiece);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    old=Path(a.grounded);m=json.loads((old/'manifest.json').read_text());allrows={};files={};mapped_total=0;costs={};unknown={}
    tokenizer=json.loads(Path(a.tokenizer_json).read_text());assert next(x['id'] for x in tokenizer['added_tokens'] if x['content']=='<extra_id_0>')==32099
    for split,key in [('train','train'),('held_input','held_input')]:
        source=Path(m['artifacts'][key]['path']);assert ref(source)==m['artifacts'][key];result=[]
        for r in rows(source):
            text='\n\n'.join(x['content'] for x in r['messages']);ids=sp.encode(text,out_type=int)+[1];assert len(ids)<=16384
            for p in sp.encode_as_immutable_proto(text).pieces:
                if p.id==2:unknown[p.surface]=unknown.get(p.surface,0)+1
            item=dict(row_id=r['row_id'],pmc_id=r['pmc_id'],patient_id=r['target_patient_id'],source_text=text,input_ids=ids,
                source_sha256=hashlib.sha256(text.encode()).hexdigest())
            if split=='train':
                labels,transport,mapped=quote_codec.encode(sp,r['answer']);mapped_total+=mapped
                item.update(target_text=r['answer'],transport_target_text=transport,labels=labels,
                    original_target_sha256=hashlib.sha256(r['answer'].encode()).hexdigest(),
                    transport_target_sha256=hashlib.sha256(transport.encode()).hexdigest(),
                    labels_sha256=hashlib.sha256(json.dumps(labels).encode()).hexdigest(),mapped_less_than=mapped)
            else:assert 'answer' not in r
            result.append(item)
        path=out/(split+'.jsonl.gz');path.write_bytes(gzip.compress(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in result).encode(),mtime=0))
        files[split]=ref(path);allrows[split]=result
        costs[split]={'rows':len(result),'encoder_total':sum(len(r['input_ids']) for r in result),'encoder_max':max(len(r['input_ids']) for r in result),
            'decoder_total':sum(len(r.get('labels',[])) for r in result),'decoder_max':max(len(r.get('labels',[])) for r in result)}
    train=allrows['train'];assert len(train)==87 and len(allrows['held_input'])==39 and mapped_total==2
    short=max((i for i,r in enumerate(train) if len(r['input_ids'])<=8192),key=lambda i:len(train[i]['input_ids']))
    long=max(range(87),key=lambda i:len(train[i]['input_ids']));target=max(range(87),key=lambda i:len(train[i]['labels']))
    cases=[dict(name='max_target_native',index=target,pad_to=None),dict(name='8k_padded',index=short,pad_to=8192),
           dict(name='max_input_native',index=long,pad_to=None),dict(name='16k_padded',index=long,pad_to=16384)]
    for c in cases:c.update(row_id=train[c['index']]['row_id'],source_tokens=len(train[c['index']]['input_ids']),target_tokens=len(train[c['index']]['labels']))
    result=dict(status='COMPLETE_FROZEN_G87_QUOTES_WITH_EXISTING_SENTINEL_TRANSPORT',source=ref(__file__),codec=ref(Path(quote_codec.__file__)),
        grounded_manifest=ref(old/'manifest.json'),grounded_counts=m['counts'],artifacts=files,counts=costs,probe_cases=cases,
        native_spiece=ref(a.spiece),model_pad_id=0,eos_id=1,decoder_start_id=0,
        target_semantics_original_G_exact_all87=True,raw_target_byte_roundtrip_exact_after_codec=True,
        original_and_transport_answer_both_saved_and_hashed=True,mapped_target_less_than=2,new_tokens_or_parameters=0,
        unicode_escape_attempt={'input':'["\\u003c"]','native_UNK':1,'roundtrip_exact':False,'reason':'backslash also absent in native SP'},
        target_transport='native SP IDs except original unknown less-than maps existing extra_id_0 ID32099; decode pieces restores less-than before original parser',
        input_native_UNK_surfaces=unknown,input_characters_unmodified=True,no_patient_drop_no_new_quote_targets=True,
        full818_relation_supervision_claimed=False,limits='Original G includes801 weak anchors/814canonical keys;4 invalid raw relation labels remain documented incomplete, no covert conversion',
        no_model_no_fit_no_held_gold_no_quality=True)
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'costs':costs,'mapped':mapped_total,'cases':cases,'inputs_UNKNOWN':unknown},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('grounded','spiece','tokenizer-json','output'):p.add_argument('--'+n,required=True)
    main(p.parse_args())
