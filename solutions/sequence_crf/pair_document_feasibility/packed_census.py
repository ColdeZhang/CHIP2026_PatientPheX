#!/usr/bin/env python3
"""Input-only packed document census. No model, labels, fitting or evaluation."""
import argparse, collections, datetime, gzip, hashlib, json, os, signal, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from tokenizers import Tokenizer
from census import ref, load, rows, save, chunks


def pack(doc, tokenizer):
    """Independent passage tokenization, explicit SEP, reversible token offset map."""
    ids, mapping, starts = [], [], []
    for pi, passage in enumerate(doc['full_text']):
        if pi:
            ids.append(tokenizer.token_to_id('[SEP]'))
            mapping.append(None)  # Encoder-visible separator; never a mention/context token.
        starts.append(len(ids))
        enc = tokenizer.encode(passage['text'], add_special_tokens=False)
        for tid, (left, right) in zip(enc.ids, enc.offsets):
            assert 0 <= left < right <= len(passage['text'])
            ids.append(tid)
            mapping.append((pi, left, right, passage['offset'] + left, passage['offset'] + right))
    assert ids and len(ids) == len(mapping)
    return ids, mapping, starts


def occurrence_tokens(doc, mapping, occurrence):
    begin, end = occurrence['offset'], occurrence['offset'] + occurrence['length']
    pi = occurrence['passage_index']
    passage = doc['full_text'][pi]
    assert passage['text'][begin-passage['offset']:end-passage['offset']] == occurrence['text']
    ts = [i for i, x in enumerate(mapping) if x is not None and x[0] == pi and x[3] < end and begin < x[4]]
    assert ts and ts == list(range(ts[0], ts[-1] + 1))
    assert all(mapping[i] is not None for i in ts)
    return ts[0], ts[-1] + 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--directory', type=Path, required=True)
    a = ap.parse_args(); b = a.directory; p = load(b/'packed-plan.json'); start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU120'))); signal.alarm(120)
    assert p['sources']['packed_census'] == ref(__file__)
    assert p['sources']['census'] == ref(Path(__file__).with_name('census.py'))
    for r in p['references'].values(): assert ref(r['path']) == r
    prior = load(p['references']['passage_report']['path'])
    for r in prior['references'].values(): assert ref(r['path']) == r
    tok = Tokenizer.from_file(prior['references']['tokenizer']['path']); tok.no_truncation(); tok.no_padding()
    old_occ = load(prior['artifacts']['occurrences']['path']); old_pairs = load(prior['artifacts']['pairs']['path'])
    old_docs = {x['pmc_id']: x for x in load(prior['artifacts']['documents']['path'])}
    occurrence_groups = collections.defaultdict(list)
    for x in old_occ: occurrence_groups[x['pmc_id']].append(x)
    pair_groups = collections.defaultdict(list)
    for x in old_pairs: pair_groups[x['pmc_id']].append(x)
    stats = collections.defaultdict(collections.Counter); docs_out = []; occ_out = []; windows_out = []
    token_out = []; pair_out = []
    for spec in prior['strict_sources']:
        assert ref(spec['input']['path']) == spec['input']
        for doc in rows(spec['input']['path']):
            assert not {'entities', 'association'} & doc.keys()
            pmc, split = str(doc['pmc_id']), spec['split']; ids, mapping, starts = pack(doc, tok)
            windows = chunks(len(ids)); cover = [0] * len(ids)
            for wi, (left, right) in enumerate(windows):
                for i in range(left, right): cover[i] += 1
                windows_out.append(dict(pmc_id=pmc, window_id=wi, start=left, end=right,
                                        input_ids=[tok.token_to_id('[CLS]')]+ids[left:right]+[tok.token_to_id('[SEP]')],
                                        pooling_mask=[False]+[x is not None for x in mapping[left:right]]+[False]))
            assert min(cover) >= 1 and len(ids) == old_docs[pmc]['raw_tokens']+len(doc['full_text'])-1
            entities = collections.defaultdict(list)
            for o in occurrence_groups[pmc]:
                left, right = occurrence_tokens(doc, mapping, o)
                assert (left, right) == (starts[o['passage_index']]+o['token_start'], starts[o['passage_index']]+o['token_end'])
                ws = [i for i, (l, r) in enumerate(windows) if l <= left < right <= r]
                assert ws, (pmc, 'no window contains full occurrence', o)
                assert mapping[left][0] == mapping[right-1][0]
                item = dict(o, stream_start=left, stream_end=right, packed_window_ids=ws)
                occ_out.append(item); entities[(o['kind'], o['key'])].append(item)
            no_common = 0
            for pair in pair_groups[pmc]:
                ps = entities[('patient', pair['patient_id'])]; cs = entities[('phenotype', pair['concept'])]
                assert ps and cs  # Unknown/missing identities hard fail; no silent zero query.
                pw = {w for x in ps for w in x['packed_window_ids']}; cw = {w for x in cs for w in x['packed_window_ids']}
                no_common += not pw.intersection(cw)
                pair_out.append(dict(pair, packed_co_windows=len(pw & cw), global_context_eligible_tokens=old_docs[pmc]['raw_tokens']))
            lens = [r-l+2 for l,r in windows]
            # Proposed encoder microbatch<=4 windows, stable within-document order; no cross-document mixing.
            padded4 = sum(max(lens[i:i+4])*len(lens[i:i+4]) for i in range(0,len(lens),4))
            raw_n = old_docs[pmc]['raw_tokens']; count_pairs = len(pair_groups[pmc])
            d = dict(split=split, pmc_id=pmc, raw_tokens=raw_n, separators=len(doc['full_text'])-1,
                     stream_tokens=len(ids), windows=len(windows), actual_encoded_tokens=sum(lens),
                     padded512_tokens=512*len(windows), dynamic4_padded_tokens=padded4,
                     max_input_tokens=max(lens), pair_count=count_pairs, pairs_no_common_window=no_common,
                     all_occurrences=len(occurrence_groups[pmc]), full_occurrence_covered=len(occurrence_groups[pmc]),
                     max_occurrence_tokens=max(x['stream_end']-x['stream_start'] for x in occ_out if x['pmc_id']==pmc),
                     pair_context_scalar_cells=count_pairs*raw_n, context_hidden_fp32_bytes=raw_n*768*4,
                     max_pairchunk32_attention_fp32_bytes=min(32,count_pairs)*raw_n*4,
                     prior_passage_windows=old_docs[pmc]['windows'], prior_passage_actual_tokens=old_docs[pmc]['window_tokens'],
                     prior_passage_padded512_tokens=old_docs[pmc]['pad512_tokens'])
            docs_out.append(d)
            for key, value in d.items():
                if isinstance(value,int) and not key.startswith('max_') and key != 'context_hidden_fp32_bytes': stats[split][key] += value
            stats[split]['documents'] += 1
            token_out.append(dict(pmc_id=pmc, input_ids=ids, offset_map=mapping, occurrence_copy_policy='Mean all overlapping encoded copies per original token; span mean then occurrence mean. Separators excluded.'))
    for name, value in [('packed-documents',docs_out),('packed-occurrences',occ_out),('packed-pairs',pair_out)]: save(b/(name+'.json'),value)
    for name, value in [('packed-token-map',token_out),('packed-windows',windows_out)]:
        path = b/(name+'.jsonl.gz'); assert not path.exists()
        with path.open('xb') as raw:
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:
                for x in value: f.write((json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n').encode())
    report = dict(status='CPU_INPUT_ONLY_NO_ENCODER_OR_TRAIN', completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  seconds=time.monotonic()-start, sources=p['sources'], plan=ref(b/'packed-plan.json'), totals=stats,
                  max_windows_per_document=max(x['windows'] for x in docs_out),
                  max_doc_context_hidden_fp32_bytes=max(x['context_hidden_fp32_bytes'] for x in docs_out),
                  max_pairchunk32_attention_fp32_bytes=max(x['max_pairchunk32_attention_fp32_bytes'] for x in docs_out),
                  max_occurrence_tokens=max(x['max_occurrence_tokens'] for x in docs_out),
                  max_input_tokens=max(x['max_input_tokens'] for x in docs_out),
                  all_occurrence_offsets_and_full_window_coverage=True, no_missing_pair_endpoint=True,
                  tokenizer_reset=True, stride_interpretation='HF stride/overlap128 => window advance382; not advance128',
                  refs=prior['references'], artifacts={x:ref(b/x) for x in ['packed-documents.json','packed-occurrences.json','packed-pairs.json','packed-token-map.jsonl.gz','packed-windows.jsonl.gz']},
                  label_reads=False, encoder_forward=False, model_parameters_loaded=False)
    save(b/'packed-report.json',report); print(json.dumps({k:report[k] for k in ['seconds','totals','max_windows_per_document','max_occurrence_tokens','max_doc_context_hidden_fp32_bytes','max_pairchunk32_attention_fp32_bytes']}))

if __name__ == '__main__': main()
