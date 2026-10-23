#!/usr/bin/env python3
"""CPU-only, one deterministic same-HPO whole-mention variant per safe train window."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from run import load_hpo
import run as ontology_source

SEED = 20261023


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def norm(value):
    return ' '.join(str(value).lower().split())


def is_no(entity):
    return str(entity.get('note') or '').upper() == 'NO'


def span(entity):
    return int(entity['offset']), int(entity['length'])


def summary(rows):
    def distribution(values):
        values = sorted(values)
        if not values:
            return {'count': 0, 'sum': 0}
        return {'count': len(values), 'sum': sum(values), 'min': values[0],
                'median': values[len(values)//2], 'p95': values[math.ceil(.95*len(values))-1],
                'max': values[-1]}
    return {'rows': len(rows), 'positive_window_spans': sum(len(r['ner']) for r in rows),
            'no_window_spans': sum(len(r['no_span_indices']) for r in rows),
            'tokens': distribution([r['tokens'] for r in rows]),
            'words': distribution([len(r['tokenized_text']) for r in rows]),
            'positive_widths': distribution([b-a+1 for r in rows for a,b,_ in r['ner']]),
            'all_negative_windows': sum(not r['ner'] for r in rows)}


def audit_row(row, processor, torch):
    """Actual word splitter, tokenizer, span mask and collator; never truncate."""
    words = list(processor.words_splitter(row['text']))
    assert [w[0] for w in words] == row['tokenized_text']
    assert [[row['window_start']+a,b-a] for _,a,b in words] == row['word_offsets']
    assert all(row['text'][a:b] == w for w,a,b in words)
    width = processor.config.max_width
    expected = {a*width+b-a for a,b,label in row['ner'] if label == 'phenotype'}
    assert len(expected) == len(row['ner'])
    assert [[row['word_offsets'][a][0],
             sum(row['word_offsets'][b])-row['word_offsets'][a][0]]
            for a,b,_ in row['ner']] == row['positive_spans']
    prefix = [processor.config.ent_token, 'phenotype', processor.config.sep_token]
    tokenized = processor.transformer_tokenizer(prefix+[w[0] for w in words],
                                                is_split_into_words=True, truncation=False,
                                                verbose=False)
    assert len(tokenized['input_ids']) == row['tokens'] <= 384
    raw = processor.collate_raw_batch([row], entity_types=[['phenotype']])
    tensors = processor.collate_fn(raw, prepare_labels=True)
    assert int(tensors['attention_mask'].sum()) == row['tokens']
    assert int(tensors['words_mask'].max()) == len(words)
    assert int(raw['seq_length'][0]) == len(words)
    labels, valid = tensors['labels'].reshape(-1), raw['span_mask'].reshape(-1).bool()
    assert set(torch.where(labels > 0)[0].tolist()) == expected
    assert not (labels[~valid] > 0).any()
    assert all(bool(valid[i]) and float(labels[i]) == 0 for i in row['no_span_indices'])
    return {'valid_spans': int(valid.sum()), 'positive_spans': len(expected),
            'negative_spans': int(valid.sum())-len(expected)}


def safe_window_entities(row, doc):
    """Reject the entire augmented window if its annotation geometry is unsafe."""
    start, end = row['window_start'], row['window_start']+len(row['text'])
    entities = {}
    for entity in doc['entities']:
        a,n = span(entity)
        if not (a < end and start < a+n):
            continue
        if not start <= a < a+n <= end:
            return None, 'partial_annotation_at_window_boundary'
        if row['text'][a-start:a+n-start] != entity['text']:
            return None, 'source_annotation_text_mismatch'
        if ';' in str(entity['identifier']):
            return None, 'compound_identifier_in_window'
        key = (a,n)
        if key in entities and (str(entities[key]['identifier']) != str(entity['identifier'])
                                or is_no(entities[key]) != is_no(entity)):
            return None, 'conflicting_duplicate_span'
        entities[key] = entity
    ordered = [entities[k] for k in sorted(entities)]
    if any(span(right)[0] < sum(span(left)) for left,right in zip(ordered,ordered[1:])):
        return None, 'overlapping_annotations_in_window'
    if {span(e) for e in ordered if not is_no(e)} != {tuple(x) for x in row['positive_spans']}:
        return None, 'positive_not_represented_in_original_window'
    word_start = {a:i for i,(a,_) in enumerate(row['word_offsets'])}
    word_end = {a+n:i for i,(a,n) in enumerate(row['word_offsets'])}
    expected_no = set()
    for entity in ordered:
        a,n = span(entity)
        if a not in word_start or a+n not in word_end:
            return None, 'annotation_not_at_original_word_boundary'
        first,last = word_start[a],word_end[a+n]
        if not 0 < last-first+1 <= 12:
            return None, 'annotation_exceeds_original_max_width'
        if is_no(entity):
            expected_no.add(first*12+last-first)
    if expected_no != set(row['no_span_indices']):
        return None, 'no_annotation_not_represented'
    return ordered, None


def make_variant(row, entities, target, replacement, processor):
    a,n = span(target)
    left,right = a-row['window_start'],a+n-row['window_start']
    text = row['text'][:left]+replacement+row['text'][right:]
    delta = len(replacement)-n
    words = list(processor.words_splitter(text))
    prefix = [processor.config.ent_token,'phenotype',processor.config.sep_token]
    tokens = len(processor.transformer_tokenizer(prefix+[w[0] for w in words],
                 is_split_into_words=True,truncation=False,verbose=False)['input_ids'])
    if tokens > 384:
        return None, 'replacement_exceeds_token_budget'
    starts = {x+row['window_start']:i for i,(_,x,_) in enumerate(words)}
    ends = {y+row['window_start']:i for i,(_,_,y) in enumerate(words)}
    mapped, positives, negatives = [], [], []
    for entity in entities:
        old_start,old_length = span(entity)
        if (old_start,old_length) == (a,n):
            new_start,new_length,surface = a,len(replacement),replacement
        else:
            new_start = old_start + (delta if old_start >= a+n else 0)
            new_length,surface = old_length,entity['text']
        assert text[new_start-row['window_start']:new_start+new_length-row['window_start']] == surface
        if new_start not in starts or new_start+new_length not in ends:
            return None, 'replacement_breaks_annotation_word_boundary'
        first,last = starts[new_start],ends[new_start+new_length]
        if not 0 < last-first+1 <= 12:
            return None, 'replacement_exceeds_span_width'
        mapped.append({'source_offset':old_start,'source_length':old_length,
                       'offset':new_start,'length':new_length,'identifier':entity['identifier'],
                       'no':is_no(entity)})
        if is_no(entity):
            negatives.append(first*12+last-first)
        else:
            positives.append((first,last,new_start,new_length))
    result = copy.deepcopy(row)
    result.update(row_id=row['row_id']+':same-hpo-aug',text=text,
                  tokenized_text=[w for w,_,_ in words],
                  word_offsets=[[row['window_start']+x,y-x] for _,x,y in words],
                  ner=[[x,y,'phenotype'] for x,y,_,_ in positives],
                  positive_spans=[[a,n] for _,_,a,n in positives],
                  no_span_indices=sorted(negatives),tokens=tokens)
    assert len(result['ner']) == len(row['ner'])
    assert len(result['no_span_indices']) == len(row['no_span_indices'])
    result['augmentation'] = {'method':'same_canonical_hpo_whole_mention_v1',
                              'source_row_id':row['row_id'],'source_offset':a,'source_length':n,
                              'canonical_hpo':target['identifier'],'original_text':target['text'],
                              'replacement':replacement,'length_delta':delta,
                              'coordinate_space':'synthetic training window, never original prediction',
                              'all_annotation_map':mapped}
    return result, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['train32','prepared32','prepared-manifest','hpo','loader-source',
                 'model-dir','tokenizer-dir','output-dir']:
        p.add_argument('--'+name,required=True)
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), 'Run with CUDA_VISIBLE_DEVICES empty; CPU preparation only'
    random.seed(SEED);torch.manual_seed(SEED)
    start = time.monotonic()
    spec = importlib.util.spec_from_file_location('frozen_gliner_loader',args.loader_source)
    loader = importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    raw_manifest = json.loads(Path(args.prepared_manifest).read_text())
    assert sha(args.train32) == raw_manifest['files']['nested_train']
    assert sha(args.prepared32) == raw_manifest['artifacts']['nested_train32']['sha256']
    docs,rows = read(args.train32),read(args.prepared32)
    by_doc = {str(d['pmc_id']):d for d in docs}
    ids = {r['pmc_id'] for r in rows}
    assert len(docs) == len(ids) == 32 and ids == set(by_doc)
    assert ids == set(raw_manifest['artifacts']['nested_train32']['pmc_ids'])
    heldout_ids = set(raw_manifest['artifacts']['nested_validation16']['pmc_ids'])
    assert len(heldout_ids) == 16 and not ids & heldout_ids
    assert len(rows) == 1150 and len({r['row_id'] for r in rows}) == len(rows)
    terms,allowed,_,version = load_hpo(args.hpo)
    dictionary_ids,training_ids = defaultdict(set),defaultdict(set)
    for hid in allowed:
        for text in [terms[hid]['name']]+terms[hid]['synonyms']:
            dictionary_ids[norm(text)].add(hid)
    for doc in docs:
        for entity in doc['entities']:
            if not is_no(entity):
                training_ids[norm(entity['text'])].update(str(entity['identifier']).split(';'))
    model = loader.load(Path(args.model_dir),Path(args.tokenizer_dir))
    processor = model.data_processor
    assert processor.config.max_width == 12 and processor.config.max_len == 384
    totals,skip = Counter(),Counter()
    added,decisions = [],[]
    for row in rows:
        doc = by_doc[row['pmc_id']]
        passage = doc['full_text'][row['passage_index']]
        begin = row['window_start']-int(passage['offset'])
        assert passage['text'][begin:begin+len(row['text'])] == row['text']
        totals.update({'original_'+k:v for k,v in audit_row(row,processor,torch).items()})
        event = {'source_row_id':row['row_id']}
        decisions.append(event)
        if not row['ner']:
            event['skip_reason']='no_positive_in_window';skip[event['skip_reason']]+=1;continue
        entities,reason = safe_window_entities(row,doc)
        if reason:
            event['skip_reason']=reason;skip[reason]+=1;continue
        options = []
        for entity in entities:
            hid = str(entity['identifier'])
            if is_no(entity) or hid not in allowed:
                continue
            for replacement in sorted(set([terms[hid]['name']]+terms[hid]['synonyms'])):
                key = norm(replacement)
                if key == norm(entity['text']) or len(replacement) <= 2:
                    continue
                if dictionary_ids[key] != {hid} or (training_ids[key] and training_ids[key] != {hid}):
                    continue
                replacement_words = list(processor.words_splitter(replacement))
                if not 0 < len(replacement_words) <= 12:
                    continue
                if not replacement_words or replacement_words[0][1] != 0 or replacement_words[-1][2] != len(replacement):
                    continue
                options.append((span(entity),hid,replacement,entity))
        options.sort(key=lambda x:(x[0],x[1],x[2]))
        if not options:
            event['skip_reason']='no_unique_same_canonical_alias';skip[event['skip_reason']]+=1;continue
        row_seed = int(hashlib.sha256((str(SEED)+':'+row['row_id']).encode()).hexdigest(),16)
        selection = random.Random(row_seed).randrange(len(options))
        _,hid,replacement,target = options[selection]
        event.update(candidate_options=len(options),selected_index=selection,
                     target_span=list(span(target)),canonical_hpo=hid,replacement=replacement)
        variant,reason = make_variant(row,entities,target,replacement,processor)
        if reason:
            event['skip_reason']=reason;skip[reason]+=1;continue
        totals.update({'augmented_'+k:v for k,v in audit_row(variant,processor,torch).items()})
        event.update(added_row_id=variant['row_id'],tokens=variant['tokens'])
        added.append(variant)
    assert len(added)+sum(skip.values()) == len(rows)
    assert len({r['augmentation']['source_row_id'] for r in added}) == len(added)
    out = Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'augmented_train32.jsonl').exists()
    original_bytes = Path(args.prepared32).read_bytes()
    assert original_bytes.endswith(b'\n')
    append_bytes = ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in added).encode()
    combined = out/'augmented_train32.jsonl'
    combined.write_bytes(original_bytes+append_bytes)
    assert combined.read_bytes()[:len(original_bytes)] == original_bytes
    assert read(combined)[:len(rows)] == rows
    (out/'added_train32.jsonl').write_bytes(append_bytes)
    (out/'decisions.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in decisions))
    original_file = out/'original_train32.jsonl';original_file.write_bytes(original_bytes)
    manifest = copy.deepcopy(raw_manifest)
    manifest['configuration'] = {'method':'same_canonical_hpo_whole_mention_v1','seed':SEED,
                                 'policy':'preserve all original rows; append at most one deterministic variant per safe window',
                                 'choice_failure_policy':'skip selected variant, no fallback candidate or text truncation',
                                 'cpu_only':True,'parent_data_manifest_sha256':sha(args.prepared_manifest)}
    manifest['artifacts']['nested_train32'] = {'path':str(combined),'rows':len(rows)+len(added),
                                              'sha256':sha(combined),'bytes':combined.stat().st_size,
                                              'pmc_ids':sorted(ids)}
    # Heldout metadata is used for membership only; no heldout gold file is read or copied.
    manifest['artifacts'] = {k:v for k,v in manifest['artifacts'].items()
                             if k in ['nested_train32','nested_validation16']}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    report = {'scope':'CPU preparation, official nested_train32 only; no heldout gold read or predictions',
              'configuration':vars(args),'seed':SEED,'hpo_version':version,
              'original':summary(rows),'added':summary(added),'combined':summary(rows+added),
              'actual_collator_counts':dict(totals),'skip_reasons':dict(skip),
              'original_rows_preserved_byte_for_byte':True,'new_rows_at_most_one_per_original':True,
              'all_added_positive_and_NO_annotations_preserved':True,'no_truncation':True,
              'cpu_only':True,'source_sha256':sha(__file__),
              'ontology_source_sha256':sha(ontology_source.__file__),
              'loader_source_sha256':sha(args.loader_source),
              'inputs':{k:sha(getattr(args,k)) for k in ['train32','prepared32','prepared_manifest','hpo']},
              'model_files':{str(p):sha(p) for p in [Path(args.model_dir)/'pytorch_model.bin',Path(args.model_dir)/'gliner_config.json',Path(args.tokenizer_dir)/'config.json',Path(args.tokenizer_dir)/'spm.model',Path(args.tokenizer_dir)/'tokenizer_config.json']},
              'training_change':{'base_steps_per_epoch':math.ceil(len(rows)/4),
                                 'augmented_steps_per_epoch':math.ceil((len(rows)+len(added))/4),
                                 'base_steps_10epochs':10*math.ceil(len(rows)/4),
                                 'augmented_steps_10epochs':10*math.ceil((len(rows)+len(added))/4),
                                 'note':'Both data content/quantity and optimizer update count change; not an equal-compute causal control.'},
              'elapsed_seconds':time.monotonic()-start,
              'outputs':{p.name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
                         for p in [combined,out/'added_train32.jsonl',out/'decisions.jsonl',original_file,out/'manifest.json']}}
    (out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'complete':True,'original':report['original'],'added':report['added'],
                      'skip_reasons':report['skip_reasons'],'steps':report['training_change'],
                      'seconds':report['elapsed_seconds']}),flush=True)


if __name__ == '__main__':
    main()
