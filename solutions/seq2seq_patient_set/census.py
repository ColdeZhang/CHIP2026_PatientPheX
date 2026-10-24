"""Native tokenizer/target census only; no weights, tensor framework, held labels or scoring."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time


def ref(path):
    path=Path(path).resolve();b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


def source(doc,pid):
    # Reuse grounded's complete passage ordering; list tables avoid T5's unsupported braces.
    patients=[]
    for patient in doc['patient']:
        mentions=[]
        for mention in patient['mention']:
            matches=[]
            for i,p in enumerate(doc['full_text']):
                local=mention['offset']-p['offset'];text=mention['text']
                if 0<=local<=len(p['text'])-len(text) and p['text'][local:local+len(text)]==text:
                    matches.append([f'U{i:04d}',local,text])
            assert len(matches)==1
            mentions.append(matches[0])
        patients.append([patient['patient_id'],mentions])
    units=[f"[U{i:04d} | {p.get('section_type','')} | {p.get('type','')}]\n{p['text']}"
           for i,p in enumerate(doc['full_text'])]
    instruction=('List all positive phenotype identifiers belonging to the TARGET patient. '
        'Use the whole report and distinguish this patient from relatives and other cases. '
        'Return a JSON array of identifier strings; return [] for an empty set.\n')
    body=(instruction+f'TARGET: {pid}\nPATIENT MENTIONS: '+json.dumps(patients,ensure_ascii=False,separators=(',',':'))+
          '\n\nFULL REPORT:\n'+'\n\n'.join(units))
    assert all(p['text'] in unit for p,unit in zip(doc['full_text'],units))
    return body


def atoms(values):
    result=set()
    for value in values:
        result.update(value.split(';') if re.fullmatch(r'HP:[0-9]{7}(;HP:[0-9]{7})+',value) else [value])
    return sorted(result)


def summary(values):
    ordered=sorted(values)
    return dict(count=len(values),total=sum(values),min=min(values),median=ordered[len(ordered)//2],
                p95=ordered[min(len(ordered)-1,int(.95*len(ordered)))],max=max(values),
                over4096=sum(n>4096 for n in values),over8192=sum(n>8192 for n in values),
                over16384=sum(n>16384 for n in values))


def main(args):
    import sentencepiece as spm
    from tokenizers import Tokenizer
    import tokenizers
    start=time.monotonic();public=Path(args.public);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    sp=spm.SentencePieceProcessor(model_file=str(public/'longt5-spiece.model'))
    tk=Tokenizer.from_file(str(public/'longt5-tokenizer.json'));tk.no_truncation();tk.no_padding()
    train=rows(args.train);held=rows(args.held_input)
    assert ref(args.train)['sha256']=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    assert ref(args.held_input)['sha256']=='962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'
    assert len(train)==32 and len(held)==16 and not {d['pmc_id'] for d in train}&{d['pmc_id'] for d in held}
    assert all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in held)
    records=[];mismatches=[];unknown=Counter();target_unknown=0;targets=0;empties=0;nonhpo=0
    for split,docs in [('train',train),('held_input',held)]:
        for doc in docs:
            assoc={a['patient_id']:atoms(a['phenotype']) for a in doc.get('association',[])}
            for patient in doc['patient']:
                pid=patient['patient_id'];text=source(doc,pid);ids=sp.encode(text,out_type=int)+[sp.eos_id()]
                fast=tk.encode(text,add_special_tokens=True)
                if ids!=fast.ids:mismatches.append(dict(pmc_id=doc['pmc_id'],patient_id=pid,kind='encoder',sp=len(ids),fast=len(fast.ids)))
                for token,offset in zip(fast.ids,fast.offsets):
                    if token==sp.unk_id():unknown[text[offset[0]:offset[1]]]+=1
                record=dict(split=split,pmc_id=doc['pmc_id'],patient_id=pid,encoder_tokens=len(ids),
                    source_characters=len(text),source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    passages=len(doc['full_text']),all_raw_passages_included=True,unknown_tokens=ids.count(sp.unk_id()))
                if split=='train':
                    values=assoc[pid];target=json.dumps(values,ensure_ascii=False,separators=(',',':'))
                    y=sp.encode(target,out_type=int)+[sp.eos_id()];decoded=sp.decode(y[:-1])
                    if y!=tk.encode(target,add_special_tokens=True).ids:mismatches.append(dict(pmc_id=doc['pmc_id'],patient_id=pid,kind='target'))
                    target_unknown+=y.count(sp.unk_id());targets+=len(values);empties+=not values
                    nonhpo+=sum(not re.fullmatch(r'HP:[0-9]{7}',v) for v in values)
                    record.update(decoder_tokens=len(y),target_atoms=len(values),target_roundtrip_exact=decoded==target,
                                  target_json_roundtrip_equal=json.loads(decoded)==values if '<unk>' not in decoded else False,
                                  target_sha256=hashlib.sha256(target.encode()).hexdigest())
                records.append(record)
    assert len([r for r in records if r['split']=='train'])==87 and len(records)==126
    result=dict(status='COMPLETE_NATIVE_SEQ2SEQ_INPUT_TARGET_CENSUS_NO_MODEL',source=ref(__file__),
        inputs={'train':ref(args.train),'held_input':ref(args.held_input)},
        tokenizer={'sentencepiece_model':ref(public/'longt5-spiece.model'),'native_json':ref(public/'longt5-tokenizer.json'),
                   'sentencepiece_version':spm.__version__,'tokenizers_version':tokenizers.__version__,
                   'eos_id':sp.eos_id(),'pad_id':sp.pad_id(),'unk_id':sp.unk_id()},
        train_encoder=summary([r['encoder_tokens'] for r in records if r['split']=='train']),
        held_encoder=summary([r['encoder_tokens'] for r in records if r['split']=='held_input']),
        train_decoder=summary([r['decoder_tokens'] for r in records if r['split']=='train']),
        total_target_atoms=targets,empty_patients=empties,non_HPO_literal_atoms=nonhpo,
        target_unknown_tokens=target_unknown,all_target_roundtrips_equal=all(r.get('target_json_roundtrip_equal',True) for r in records),
        native_ID_mismatches=mismatches,unknown_source_tokens=sum(r['unknown_tokens'] for r in records),
        unknown_source_codepoints=dict(unknown),full_passage_raw_strings_preserved_before_native_tokenization=True,
        no_truncation_no_chunks_no_entity_labels_in_input=True,no_held_gold_no_model_no_fit_no_quality=True,
        seconds=time.monotonic()-start)
    (out/'rows.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    result['rows']=ref(out/'rows.json');(out/'complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','tokenizer','unknown_source_codepoints','source','rows')},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for n in ('public','train','held-input','output'):ap.add_argument('--'+n,required=True)
    main(ap.parse_args())
