"""CPU-only native Task1 target audit; never constructs a neural model or optimizer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from prepare import build

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--model-assets',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();started=time.monotonic()
    import torch
    from tokenizers import Tokenizer
    from gliner2.models.base import load_extractor_tokenizer
    from gliner2.processor import SchemaTransformer
    from gliner2.inference.chunking import split_text_into_chunks
    from gliner2.models.boundary.targets_device import dense_targets_from_pairs
    assert not torch.cuda.is_available();torch.set_num_threads(1)
    assert sha(args.train)=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    processor=SchemaTransformer(tokenizer=load_extractor_tokenizer(args.model_assets));native=Tokenizer.from_file(str(Path(args.model_assets)/'tokenizer.json'));cache={}
    docs=[json.loads(x) for x in Path(args.train).read_text().splitlines() if x];assert len(docs)==32
    counts=Counter();rows=[];unrepresentable=[];audits=[]
    for raw in docs:
        doc=build({k:v for k,v in raw.items() if k not in ('entities','association')});tokens=list(processor.word_splitter(doc['text'],lower=False));starts={a for t,a,b in tokens};ends={b for t,a,b in tokens};ann=[]
        for index,e in enumerate(raw['entities']):
            no=str(e.get('note') or '').upper()=='NO';found=[]
            for p in doc['passages']:
                a=e['offset']-p['offset']+p['start'];b=a+e['length']
                if p['start']<=a<b<=p['end'] and doc['text'][a:b]==e['text']:found.append((a,b))
            assert len(found)==1,(doc['pmc_id'],index);a,b=found[0];aligned=a in starts and b in ends
            item=dict(annotation_index=index,start=a,end=b,text=e['text'],no=no,aligned=aligned,original_offset=e['offset']);ann.append(item);counts['NO_annotations' if no else 'positive_annotations']+=1
            if not aligned:unrepresentable.append(dict(pmc_id=doc['pmc_id'],**item));counts['unaligned_NO' if no else 'unaligned_positive']+=1
        positive={(x['start'],x['end']) for x in ann if not x['no'] and x['aligned']};negative={(x['start'],x['end']) for x in ann if x['no'] and x['aligned']};assert not positive&negative
        counts['unique_representable_positive']+=len(positive);counts['unique_representable_NO']+=len(negative)
        counts['positive_proper_nested_pairs']+=sum(a<=c<d<=b and (a,b)!=(c,d) for a,b in positive for c,d in positive)
        counts['positive_crossing_pairs']+=sum(a<c<b<d for a,b in positive for c,d in positive)
        chunks=split_text_into_chunks(doc['text'],384,64,processor.word_splitter);assert chunks[0].start_word==0 and chunks[-1].end_word==len(tokens);assert all(a.end_word>=b.start_word for a,b in zip(chunks,chunks[1:]));covered=set()
        for i,c in enumerate(chunks):
            selected=sorted((a-c.start_char,b-c.start_char) for a,b in positive if c.start_char<=a<b<=c.end_char);no_spans=sorted((a-c.start_char,b-c.start_char) for a,b in negative if c.start_char<=a<b<=c.end_char)
            covered.update((a+c.start_char,b+c.start_char) for a,b in selected)
            spans=[dict(text=c.text[a:b],start=a,end=b) for a,b in selected];schema=dict(entities={'phenotype':spans});tokens_c=list(processor.word_splitter(c.text,lower=False));sm={a:i for i,(t,a,b) in enumerate(tokens_c)};em={b:i+1 for i,(t,a,b) in enumerate(tokens_c)};expected=sorted((sm[a],em[b]) for a,b in selected);assert len(expected)<=64
            counts['windows']+=1;counts['positive_target_window_instances']+=len(selected);counts['NO_window_instances']+=len(no_spans);counts['empty_positive_windows']+=not selected;counts['windows_with_NO']+=bool(no_spans)
            rows.append(dict(input=c.text,output=schema));audits.append(dict(pmc_id=doc['pmc_id'],window=i,start=c.start_char,end=c.end_char,start_word=c.start_word,end_word=c.end_word,positive_chars=selected,positive_token_pairs=expected,NO_chars=no_spans,unrepresentable_annotations=[x['annotation_index'] for x in ann if not x['aligned'] and c.start_char<=x['start']<x['end']<=c.end_char]))
        assert covered==positive;counts['documents']+=1
    assert counts['windows']==320 and len(rows)%2==0
    (out/'prepared-summary.json').write_text(json.dumps(dict(counts=dict(counts),rows=len(rows),unrepresentable=unrepresentable),indent=2)+'\n')
    (out/'train.native.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    max_ids=max_targets=0;native_periods=0
    for start in range(0,len(rows),2):
        group=rows[start:start+2];batch=processor.collate_fn_train([(x['input'],x['output']) for x in group],max_len=None,error_policy='raise',architecture='boundary',max_gold_per_query=64,on_capacity_exceeded='raise',dedupe_contained_entity_spans=False)
        assert batch.input_ids.shape[0]==len(group)==2 and batch.query_marker_mask.sum().item()==2
        assert batch.targets.start_targets is batch.targets.end_targets is batch.targets.inside_targets is None
        # Same lazy native target construction used by BoundaryHead._compute_losses.
        text_width=batch.text_word_mask.shape[-1]
        dense_start,dense_end,dense_inside=dense_targets_from_pairs(batch.targets.mention_pairs.clamp(0,text_width),batch.targets.mention_mask,text_width)
        for j,row in enumerate(group):
            audit=audits[start+j];actual=batch.targets.mention_pairs[j,0][batch.targets.mention_mask[j,0]].tolist();assert sorted(map(tuple,actual))==list(map(tuple,audit['positive_token_pairs']))
            assert int(batch.targets.mention_mask[j].sum())==len(actual);max_targets=max(max_targets,len(actual))
            assert not batch.targets.mention_mask[j,0,len(actual):].any()
            nwords=batch.text_word_counts[j];assert int(batch.text_word_mask[j].sum())==nwords and not batch.text_word_mask[j,nwords:].any()
            assert batch.original_texts[j] in (row['input'],row['input']+'.');native_periods+=batch.original_texts[j]!=row['input']
            expected_starts={a for a,b in audit['positive_token_pairs']};expected_ends={b for a,b in audit['positive_token_pairs']};expected_inside={i for a,b in audit['positive_token_pairs'] for i in range(a,b)}
            assert set(dense_start[j,0].nonzero().flatten().tolist())==expected_starts
            assert set(dense_end[j,0].nonzero().flatten().tolist())==expected_ends
            assert set(dense_inside[j,0].nonzero().flatten().tolist())==expected_inside
            combined=[]
            for item in batch.schema_tokens_list[j]:combined.extend(item);combined.append(processor.SEP_STRUCT)
            if combined:combined.pop()
            combined.append(processor.SEP_TEXT);combined.extend(batch.text_tokens[j]);expected_ids=[]
            for token in combined:
                if token not in cache:cache[token]=native.encode(token,add_special_tokens=False).ids
                expected_ids.extend(cache[token])
            n=int(batch.attention_mask[j].sum());assert not batch.attention_mask[j,n:].any();assert batch.input_ids[j,:n].tolist()==expected_ids;assert n<=4096;max_ids=max(max_ids,n)
            audit.update(actual_target_pairs=actual,actual_input_ids=expected_ids,query_mask=True,text_words=nwords,input_ids=n,padding_and_dense_targets_exact=True)
    for name,values in [('train.native.jsonl',rows),('target-audit.jsonl',audits)]:
        (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in values))
    (out/'unrepresentable.json').write_text(json.dumps(unrepresentable,ensure_ascii=False,indent=2)+'\n')
    report=dict(status='CPU_ALL_TRAIN32_NATIVE_TASK1_TARGETS_PASS_NO_FIT',counts=dict(counts),max_input_ids=max_ids,max_gold_per_query_observed=max_targets,native_appended_period_windows=native_periods,batch_size=2,training_steps_per_epoch=len(rows)//2,proposed_epochs=10,proposed_optimizer_steps=10*len(rows)//2,tail_samples_dropped=0,all_representable_targets_exact=True,all_dense_start_end_inside_and_padding_masks_verified=True,all_tokenizer_json_input_ids_exact=True,silent_truncation=False,NO_policy='Positive phenotype query only. NO spans are contextual negatives; no separate class or forced negative pair injection. Sparse pair-loss receives only proposed candidates.',unaligned_policy='Retain exact original annotations in audit; 38 inside-native-word positives cannot be emitted by this unchanged native splitter and are not expanded or silently relabeled.',source_sha256=sha(__file__),train_sha256=sha(args.train),model_parameters_loaded=0,forward_calls=0,backward_calls=0,optimizer_steps=0,quality_evaluations=0,elapsed_seconds=time.monotonic()-started,outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()})
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
    try:main()
    except BaseException:
        import sys,traceback
        if '--output' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--output')+1]);p.mkdir(parents=True,exist_ok=True);(p/'failure.txt').write_text(traceback.format_exc())
        raise
