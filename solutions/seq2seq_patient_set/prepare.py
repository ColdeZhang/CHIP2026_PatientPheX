"""Freeze full 87 train / 39 input-only records with original SentencePiece IDs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from census import atoms,ref,rows,source


def main(a):
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=a.spiece)
    assert sp.eos_id()==1 and sp.unk_id()==2 and sp.piece_to_id('<pad>')==0
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);artifacts={};counts={};allrows={}
    for split,path in [('train',a.train),('held_input',a.held_input)]:
        prepared=[]
        for doc in rows(path):
            assoc={r['patient_id']:atoms(r['phenotype']) for r in doc.get('association',[])}
            for p in doc['patient']:
                text=source(doc,p['patient_id']);ids=sp.encode(text,out_type=int)+[1]
                assert len(ids)<=16384
                item=dict(row_id=f"{doc['pmc_id']}:{p['patient_id']}",pmc_id=doc['pmc_id'],patient_id=p['patient_id'],
                    source_text=text,input_ids=ids,source_sha256=hashlib.sha256(text.encode()).hexdigest())
                if split=='train':
                    answer=json.dumps(assoc[p['patient_id']],ensure_ascii=False,separators=(',',':'))
                    labels=sp.encode(answer,out_type=int)+[1]
                    assert 2 not in labels and sp.decode(labels[:-1])==answer
                    item.update(target_text=answer,labels=labels,target_sha256=hashlib.sha256(answer.encode()).hexdigest())
                prepared.append(item)
        data=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in prepared).encode()
        target=out/(split+'.jsonl.gz');target.write_bytes(gzip.compress(data,mtime=0));artifacts[split]=ref(target)
        counts[split]=dict(rows=len(prepared),encoder_tokens=sum(len(r['input_ids']) for r in prepared),
                           max_encoder=max(len(r['input_ids']) for r in prepared))
        allrows[split]=prepared
    assert counts['train']['rows']==87 and counts['held_input']['rows']==39
    assert not {r['pmc_id'] for r in allrows['train']}&{r['pmc_id'] for r in allrows['held_input']}
    train=allrows['train'];short=max((i for i,r in enumerate(train) if len(r['input_ids'])<=8192),key=lambda i:len(train[i]['input_ids']))
    long=max(range(87),key=lambda i:len(train[i]['input_ids']));target=max(range(87),key=lambda i:len(train[i]['labels']))
    cases=[dict(name='max_target_native',index=target,pad_to=None),dict(name='8k_padded',index=short,pad_to=8192),
           dict(name='max_input_native',index=long,pad_to=None),dict(name='16k_padded',index=long,pad_to=16384)]
    for c in cases:c.update(row_id=train[c['index']]['row_id'],source_tokens=len(train[c['index']]['input_ids']),target_tokens=len(train[c['index']]['labels']))
    manifest=dict(status='COMPLETE_FULL87_39_NATIVE_SENTENCEPIECE_SUPPLY',source=ref(__file__),census_source=ref(Path(__file__).with_name('census.py')),
        inputs={'train':ref(a.train),'held_input':ref(a.held_input),'spiece':ref(a.spiece)},artifacts=artifacts,counts=counts,
        full_original_passages=True,supervision='all818 original publicv2 relation atoms, no HPO/NIL/literal filtering; deterministic JSON list bytes',
        train_target_bytes_roundtrip_exact=True,held_has_no_labels=True,probe_cases=cases,
        probe_padding='right pad0 with attention_mask0; no source truncation, no synthetic clinical content',
        model_pad_id=0,SP_pad_id=sp.pad_id(),eos_id=1,decoder_start_id=0,unknown_id=2,
        no_model_no_quality_no_held_gold=True)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'artifacts':artifacts,'counts':counts,'cases':cases},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('train','held-input','spiece','output'):p.add_argument('--'+n,required=True)
    main(p.parse_args())
