#!/usr/bin/env python3
"""CPU/tokenizer-only accounting for fixed target windows plus bounded context.
No model, predictions, heldout labels, new training samples, or fitting.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import time

PATTERN = re.compile(r'\w+(?:[-_]\w+)*|\S')
EXPECTED = {
    'train': '38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a',
    'rows': 'a76db949c670f21909031ac99ac42e91d2eb9a959f4ae0d6cec9ed08f7154ed0',
    'tokenizer': 'ee028763434d18611c1c36356ea1d050e90a9fa94ede57fac48b39f85f818ad1',
    'config': 'f747cc67847f8b5412dca016377cb1a8c1642c802657dfb1949aca4d22354b76',
}

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def norm(s): return ' '.join(s.casefold().split())
def distribution(xs):
    ys = sorted(xs)
    if not ys: return {'n': 0}
    return {'n': len(ys), 'sum': sum(ys), 'min': ys[0], 'median': ys[len(ys)//2],
            'p95': ys[min(len(ys)-1, int(.95*len(ys)))], 'max': ys[-1],
            'mean': sum(ys)/len(ys)}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ['train', 'rows', 'tokenizer', 'config', 'output-dir']: ap.add_argument('--'+key, required=True)
    a = ap.parse_args(); start = time.monotonic()
    for key, expected in EXPECTED.items(): assert sha(getattr(a, key)) == expected, key
    import tokenizers
    tok = tokenizers.Tokenizer.from_file(a.tokenizer)
    tok.no_truncation(); tok.no_padding()
    cfg = json.loads(Path(a.config).read_text()); prefix = [cfg['ent_token'], 'phenotype', cfg['sep_token']]
    assert cfg['max_width'] == 12 and cfg['words_splitter_type'] == 'whitespace'
    assert tok.token_to_id('[SEP]') == 2 and tok.get_vocab_size() == 128003
    docs, rows = read(a.train), read(a.rows)
    assert len(docs) == len({str(d['pmc_id']) for d in docs}) == 32 and len(rows) == 1150
    docs_by = {str(d['pmc_id']): d for d in docs}
    # Input-only projection is the sole source of context construction.
    inputs = {k: {'pmc_id': k, 'full_text': d['full_text']} for k, d in docs_by.items()}
    cache = {}
    def encode(words): return tok.encode(prefix + words, is_pretokenized=True)
    def cost(word):
        if word not in cache: cache[word] = len(tok.encode([word], is_pretokenized=True, add_special_tokens=False).ids)
        return cache[word]
    words_by = {}
    for key, d in inputs.items():
        assert 'entities' not in d and 'association' not in d
        for pi, p in enumerate(d['full_text']):
            words_by[key, pi] = [(m.group(), int(p['offset'])+m.start(), m.end()-m.start()) for m in PATTERN.finditer(p['text'])]
    # Reconstruct all original targets using label-free 384-subword/12-word-overlap policy.
    rebuilt = {}
    for key, d in inputs.items():
        for pi, p in enumerate(d['full_text']):
            ws = words_by[key, pi]; first = 0; wi = 0
            while first < len(ws):
                lo, hi, best = first+1, min(len(ws), first+384), None
                while lo <= hi:
                    end = (lo+hi)//2; count = len(encode([w[0] for w in ws[first:end]]).ids)
                    if count <= 384: best = (end, count); lo = end+1
                    else: hi = end-1
                assert best is not None
                end, count = best; chunk = ws[first:end]
                rebuilt[f'{key}:{pi}:{wi}'] = {'word_offsets': [[w[1], w[2]] for w in chunk], 'tokenized_text': [w[0] for w in chunk], 'tokens': count, 'first': first, 'last': end}
                if end == len(ws): break
                first = max(first+1, end-12); wi += 1
    assert set(rebuilt) == {r['row_id'] for r in rows}
    for r in rows:
        rec = rebuilt[r['row_id']]
        assert all(rec[k] == r[k] for k in ['word_offsets', 'tokenized_text', 'tokens'])
    def side_select(candidates, budget, left):
        selected, used = [], 1  # Existing [SEP] delimiter consumes one subword if side exists.
        for w in reversed(candidates) if left else candidates:
            n = cost(w[0])
            if used+n > budget: break
            selected.append(w); used += n
        if left: selected.reverse()
        return selected, used if selected else 0
    # Label-free, deterministic policy; no label is used to add/drop context.
    def make_context(r):
        key, pi = r['pmc_id'], r['passage_index']; d = inputs[key]; rec = rebuilt[r['row_id']]
        ws = words_by[key, pi]; first, last = rec['first'], rec['last']
        left_pi = pi if first else pi-1
        right_pi = pi if last < len(ws) else pi+1
        left = ws[:first] if first else words_by.get((key, left_pi), [])
        right = ws[last:] if last < len(ws) else words_by.get((key, right_pi), [])
        spare = 384-r['tokens']; lb = min(64, spare//2); rb = min(64, spare-spare//2)
        lw, lc = side_select(left, lb, True); rw, rc = side_select(right, rb, False)
        combined = ([w[0] for w in lw]+['[SEP]'] if lw else []) + r['tokenized_text'] + (['[SEP]']+[w[0] for w in rw] if rw else [])
        first_target = len(prefix)+(len(lw)+1 if lw else 0)
        e = encode(combined); n = len(r['tokenized_text'])
        assert len(e.ids) == r['tokens']+lc+rc <= 384
        # Proposed low-level encoder map: context is attended but never a text word.
        mask, seen = [], set()
        for word in e.word_ids:
            label = 0
            if word is not None and first_target <= word < first_target+n and word not in seen:
                label = word-first_target+1
            mask.append(label); seen.add(word)
        assert sorted(x for x in mask if x) == list(range(1, n+1))
        baseline = encode(r['tokenized_text'])
        for i in range(n):
            assert [v for v, wid in zip(e.ids, e.word_ids) if wid == first_target+i] == [v for v, wid in zip(baseline.ids, baseline.word_ids) if wid == len(prefix)+i]
        valid = {i*12+width-1 for i in range(n) for width in range(1, min(12, n-i)+1)}
        return {'encoded_tokens': len(e.ids), 'left_words': lw, 'right_words': rw, 'left_pi': left_pi, 'right_pi': right_pi, 'left_tokens': lc, 'right_tokens': rc, 'available_left': bool(left), 'available_right': bool(right), 'valid_target_spans': len(valid), 'target_words': n, 'target_word_mask_count': sum(x>0 for x in mask)}
    # Only aggregate training annotations below; none affect the preceding input policy.
    forms = defaultdict(set); annotations = {}
    for key, d in docs_by.items():
        annotations[key] = []
        for ent in d['entities']:
            label = 'NO' if str(ent.get('note') or '').upper() == 'NO' else 'positive'
            forms[norm(ent['text'])].add(label)
            annotations[key].append((int(ent['offset']), int(ent['length']), label, norm(ent['text'])))
    ambiguous = {form for form, labels in forms.items() if len(labels) == 2}
    counts = Counter(); records = []; per_section = Counter(); context_positive_unique = set()
    for r in rows:
        c = make_context(r); key, pi = r['pmc_id'], r['passage_index']; target_passage = inputs[key]['full_text'][pi]
        added = c['encoded_tokens']-r['tokens']; n = c['target_words']; positive = r['ner']; no = r['no_span_indices']
        infer_row = {k:r[k] for k in ['row_id','pmc_id','passage_index','tokens','tokenized_text']}
        assert make_context(infer_row) == c
        valid = {i*12+width-1 for i in range(n) for width in range(1,min(12,n-i)+1)}
        positive_indices = {s*12+end-s for s,end,_ in positive}
        assert positive_indices <= valid and set(no) <= valid and not positive_indices & set(no)
        for s,end,label in positive:
            assert 0 <= s <= end < n and end-s < 12 and label=='phenotype'
            start_offset,end_offset=r['word_offsets'][s][0],sum(r['word_offsets'][end])
            assert [start_offset,end_offset-start_offset] in r['positive_spans']
        edge = sum(s < 12 or n-1-end < 12 for s, end, _ in positive)
        stat = {'row_id': r['row_id'], 'target_tokens': r['tokens'], 'encoded_tokens': c['encoded_tokens'], 'added_tokens': added, 'left_tokens': c['left_tokens'], 'right_tokens': c['right_tokens'], 'target_words': n, 'positive_instances': len(positive), 'NO_instances': len(no), 'valid_target_spans': c['valid_target_spans'], 'edge_positive_instances_within12words': edge}
        counts.update(rows=1, positive_instances=len(positive), NO_instances=len(no), valid_target_spans=c['valid_target_spans'], edge_positive_instances=edge)
        counts['rows_with_context'] += added>0; counts['positive_rows_with_context'] += bool(positive) and added>0
        counts['rows_without_context'] += added==0; counts['rows_no_spare_at384'] += r['tokens']==384
        counts['rows_context_available_but_no_complete_word_fits'] += added==0 and (c['available_left'] or c['available_right'])
        counts['rows_no_context_available'] += not(c['available_left'] or c['available_right'])
        counts['positive_instances_with_context'] += len(positive) if added else 0
        counts['edge_positive_instances_with_context'] += edge if added else 0
        target_spans = {tuple(x) for x in r['positive_spans']}
        for idx in no:
            s, w = divmod(idx, 12); target_spans.add((r['word_offsets'][s][0], sum(r['word_offsets'][s+w])-r['word_offsets'][s][0]))
        stat['ambiguous_positive_NO_surface_instances'] = sum(form in ambiguous for off, length, label, form in annotations[key] if (off, length) in target_spans)
        counts['ambiguous_surface_rows'] += stat['ambiguous_positive_NO_surface_instances']>0
        counts['ambiguous_surface_rows_with_context'] += stat['ambiguous_positive_NO_surface_instances']>0 and added>0
        crossing = False; cross_type = False; partial = full = no_context = 0
        for side in ['left', 'right']:
            ws, spi = c[side+'_words'], c[side+'_pi']
            if not ws: continue
            p = inputs[key]['full_text'][spi]
            crossing |= str(p.get('section_type')) != str(target_passage.get('section_type'))
            cross_type |= str(p.get('type')) != str(target_passage.get('type'))
            counts['context_sides_same_passage' if spi==pi else 'context_sides_adjacent_passage'] += 1
            lo, hi = ws[0][1], ws[-1][1]+ws[-1][2]
            for off, length, label, form in annotations[key]:
                if off < hi and off+length > lo:
                    if label == 'positive':
                        if lo <= off and off+length <= hi: full += 1; context_positive_unique.add((key, off, length))
                        else: partial += 1
                    else: no_context += 1
        stat.update(cross_section=crossing, cross_passage_type=cross_type, context_positive_full=full, context_positive_partial=partial, context_NO_overlapping=no_context)
        counts['rows_cross_section'] += crossing; counts['rows_cross_passage_type'] += cross_type
        counts['rows_context_contains_known_positive'] += full>0; counts['context_positive_full_instances'] += full
        counts['context_positive_partial_instances'] += partial; counts['context_NO_overlap_instances'] += no_context
        per_section[str(target_passage.get('section_type'))] += added>0
        records.append(stat)
    old_lens = [r['tokens'] for r in rows]; new_lens = [r['encoded_tokens'] for r in records]
    proxy = Counter()
    for epoch in range(10):
        order = list(range(len(rows))); random.Random(20261023+epoch).shuffle(order)
        for off in range(0, len(order), 4):
            batch = order[off:off+4]; old_max = max(old_lens[i] for i in batch); new_max = max(new_lens[i] for i in batch)
            proxy['batches'] += 1
            proxy['old_padded_tokens'] += len(batch)*old_max; proxy['new_padded_tokens'] += len(batch)*new_max
            proxy['old_attention_proxy'] += len(batch)*old_max*old_max; proxy['new_attention_proxy'] += len(batch)*new_max*new_max
    assert proxy['batches']==2880
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    record_path = out/'window-statistics.jsonl'; assert not record_path.exists()
    record_path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records))
    report = {'scope': 'training32 CPU/tokenizer feasibility only; no model/collator forward, predictions, heldout gold or generated training JSONL', 'source_sha256': sha(__file__), 'inputs': {k: {'path': getattr(a,k), 'sha256':sha(getattr(a,k))} for k in EXPECTED}, 'tokenizers_version': tokenizers.__version__, 'vocab_size_unchanged': tok.get_vocab_size()==128003,
              'policy': {'target_windows': 'all original1150 unchanged; label-free reconstruction exact', 'total_actual_subword_budget':384, 'left_right_budget':64, 'budget_includes_existing_SEP_delimiters':True, 'short_spare_allocation':'left=floor(spare/2),right=ceil(spare/2),each capped64; do not redistribute unused side', 'source':'same passage outside target first, else immediate previous/next passage within same document; no farther search or section/gold filter', 'target_word_mask':'encoder attends all; map only first subword of original target words to1..N; all context/delimiters/prefix/specials0', 'supervision':'unchanged original target span_idx/labels/NO/valid-span denominator; no context or boundary-crossing candidate enters span head', 'decode':'must use low-level target-only span outputs plus original word_offsets; never predict_entities then filter'},
              'counts':dict(counts), 'ambiguous_positive_NO_surface_types':len(ambiguous), 'context_unique_known_positive_spans':len(context_positive_unique), 'target_tokens':distribution(old_lens), 'encoded_tokens':distribution(new_lens), 'added_tokens_all':distribution([r['added_tokens'] for r in records]), 'added_tokens_when_nonzero':distribution([r['added_tokens'] for r in records if r['added_tokens']]), 'cross_section_fraction_all_rows':counts['rows_cross_section']/len(rows), 'context_fraction_all_rows':counts['rows_with_context']/len(rows), 'rows_with_context_by_target_section':dict(per_section), 'compute_proxy':dict(proxy)|{'padded_linear_ratio':proxy['new_padded_tokens']/proxy['old_padded_tokens'],'padded_attention_quadratic_ratio':proxy['new_attention_proxy']/proxy['old_attention_proxy'],'not_GPU_time_or_memory_measurement':True,'target_span_head_work_unchanged':True},
              'checks':{'all1150_target_windows_rebuilt_from_label_free_fulltext_exact':True,'actual_target_token_ID_groups_preserved_all_words':True,'all_actual384_budget_and_complete_word_boundaries_pass':True,'all2236_positive_and87_NO_indices_remain_in_original_target_support':True,'valid_target_span_denominator_unchanged_by_proposed_map':True,'context_policy_does_not_read_labels':True,'no_new_token_or_embedding':True,'integrated_modified_gliner_collator_or_forward_executed':False},
              'limits':['Context availability and ambiguous annotation counts are not evidence that context fixes real FN or improves Score.','This preserves targets but changes encoder attention and average sequence lengths; no causal benefit or GPU throughput claim.','Cross-section and partial context annotations are reported without label-based context trimming.','Context has no direct supervised target; known positive context demonstrates why treating all unlabelled appended spans as negative is invalid.','The target-only words_mask is a CPU mapping contract; actual model adapter/collator/low-level decode integration remains a separate gate before any model stage.','Same document boundaries and word_offsets preserve input scope; no outside patient facts or annotation sources used.'],
              'row_statistics':{'path':str(record_path),'sha256':sha(record_path),'bytes':record_path.stat().st_size}, 'seconds':time.monotonic()-start}
    assert counts['positive_instances']==2236 and counts['NO_instances']==87
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['counts','ambiguous_positive_NO_surface_types','target_tokens','encoded_tokens','added_tokens_when_nonzero','compute_proxy','seconds']},ensure_ascii=False))

if __name__=='__main__':main()
