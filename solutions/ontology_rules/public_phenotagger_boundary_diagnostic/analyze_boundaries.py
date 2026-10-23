"""Read-only train64 boundary-supply diagnostic; never emits model predictions."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def checked_rows(spec):
    h, count = hashlib.sha256(), 0
    with Path(spec['path']).open('rb') as stream:
        for line in stream:
            h.update(line)
            count += 1
            yield json.loads(line)
    assert h.hexdigest() == spec['sha256'], spec['path']
    assert Path(spec['path']).stat().st_size == spec['bytes'], spec['path']
    assert count == spec['rows'], (spec['path'], count)


def nil_supported(supply, gold):
    # v2 consumes linked exact atoms before a remaining NIL can use one leftover.
    return '-1' in supply or bool(supply - (set(gold) - {'-1'}))


def correct(supply, query):
    return (nil_supported(supply, query['gold_span_atomic_ids'])
            if query['gold_literal_id'] == '-1' else query['gold_literal_id'] in supply)


def normalized_full_text(text):
    # Native restore_index's length-preserving normalization must precede slicing.
    return text.replace('``', '" ').replace("''", '" ').lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    signal.alarm(115)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert all(os.environ.get(k) == '0' for k in ('USE_TORCH', 'USE_TF', 'USE_FLAX'))
    assert not nil_supported({'H1'}, {'H1', '-1'})
    assert nil_supported({'H1', 'H2'}, {'H1', '-1'})
    assert nil_supported({'-1'}, {'H1', '-1'})
    assert not correct({'HP:CURRENT'}, {'gold_literal_id': 'HP:OLD', 'gold_span_atomic_ids': ['HP:OLD']})
    plan = json.loads(Path(args.plan).read_text())
    assert sha(__file__) == plan['source_sha256']
    out = Path(plan['output_directory'])
    out.mkdir(exist_ok=False)
    root = Path(plan['public_resource_root'])
    for rel, digest in plan['resources'].items():
        assert sha(root / rel) == digest, rel
    for spec in plan.get('extra_resources', []):
        assert sha(spec['path']) == spec['sha256'], spec['path']
    assert sha(plan['queries']['path']) == plan['queries']['sha256']
    queries = json.loads(Path(plan['queries']['path']).read_text())['queries']
    assert len(queries) == 384 and [q['query_index'] for q in queries] == list(range(384))
    docs = {str(d['pmc_id']): d for d in map(json.loads, (root / 'train64-cache/train64.input.jsonl').read_text().splitlines())}
    assert len(docs) == 64 and all(set(d) == {'pmc_id', 'pmid', 'patient', 'full_text'} for d in docs.values())
    qp, span_queries = defaultdict(list), defaultdict(list)
    for q in queries:
        passage = docs[q['pmc_id']]['full_text'][q['passage_index']]
        assert q['offset'] == passage['offset'] + q['local_start']
        assert q['length'] == q['local_end'] - q['local_start']
        assert passage['text'][q['local_start']:q['local_end']] == q['text']
        qp[q['pmc_id'], q['passage_index']].append(q)
        span_queries[q['pmc_id'], q['offset'], q['length']].append(q['query_index'])
    assert len(span_queries) == 333
    labels = (root / 'lable.vocab').read_text().strip().splitlines()
    assert len(labels) == 17266 and sha(plan['bioformer_label_file']) == sha(root / 'lable.vocab')
    head = json.loads((root / 'original-column-to-current-HPO.json').read_text())['mapping']
    assert [r['original_label'] for r in head] == labels
    cmap = {r['original_label']: r['current_label'] for r in head}

    def cells_checked(cells):
        assert len(cells) == 3
        for cell in cells:
            assert labels[cell['column']] == cell['hpo_id']
            assert math.isfinite(cell['probability']) and 0 <= cell['probability'] <= 1
        assert all(cells[j]['probability'] >= cells[j + 1]['probability'] for j in range(2))

    # Tokenization and generator are the exact previously executed native sources.
    # Only query-containing passages are replayed; raw caches are fully streamed.
    sys.path.insert(0, str(root / 'native-pipeline/src'))
    import nltk
    nltk.data.path = [str(root / 'nltk_data')]
    from ssplit_tokenzier import ssplit_token_pos_lemma
    from restore_index import restore_index_nest_fn
    import ml_ner
    namespace = dict(ml_ner.__dict__)
    fixed = root / 'native-pipeline/fixed_candidate_function.py'
    exec(compile(fixed.read_text(), str(fixed), 'exec'), namespace)
    build = namespace['build_ngram_testset_filted']
    items = [dict(query_index=q['query_index'], mechanism=q['mechanism'],
                  gold_kind='unlinked' if q['gold_literal_id'] == '-1' else 'linked',
                  gold_same_span_atomic_count=len(q['gold_span_atomic_ids']),
                  raw_rows={}, tuple_cache={}, canonical_rows={}, tokenizers={}) for q in queries]
    generated, query_tuples = {}, {}
    generator_started = time.monotonic()
    verified_tokens = generated_count = 0
    for (did, pi), pqueries in qp.items():
        text = docs[did]['full_text'][pi]['text']
        conll = ssplit_token_pos_lemma(text)
        sentences = [[line.split('\t') for line in part.splitlines()] for part in conll.strip().split('\n\n')]
        annotation = ''
        for sentence in sentences:
            words = [row[0] for row in sentence]
            annotation += ' '.join(words) + '\n'
            annotation += ''.join(f'{j}\t{j}\t{word}\tHP:None\t0.0\n' for j, word in enumerate(words)) + '\n'
        locations = restore_index_nest_fn(text, annotation)
        assert len(locations) == sum(map(len, sentences))
        normalized = normalized_full_text(text)
        locs, at = [], 0
        for sentence in sentences:
            sl = [(int(row[0]), int(row[1])) for row in locations[at:at + len(sentence)]]
            for row, (left, right) in zip(sentence, sl):
                assert 0 <= left < right <= len(text)
                expected = row[0].lower().replace('``', '"').replace("''", '"')
                assert normalized[left:right] == expected, (did, pi, left, right)
            locs.append(sl)
            at += len(sentence)
        verified_tokens += at
        ml_input, _, indices = build(conll)
        instances = [x for x in ml_ner.ml_intext_fn(ml_input)[0] if x]
        assert len(instances) == len(indices)
        expected = {}
        for i, instance in enumerate(instances):
            sentence, left, right = indices[str(i + 1)]
            a, b = locs[sentence][left][0], locs[sentence][right][1]
            key = (a, b, sentence, left, right)
            assert key not in expected
            expected[key] = tuple(x[0].lower() for x in instance)
        generated[did, pi] = expected
        generated_count += len(expected)
        for q in pqueries:
            item = items[q['query_index']]
            aligned = [(si, lo, hi) for si, sent in enumerate(locs)
                       for lo, (left, _) in enumerate(sent) if left == q['local_start']
                       for hi in range(lo, len(sent)) if sent[hi][1] == q['local_end']]
            assert len(aligned) <= 1
            if aligned:
                si, lo, hi = aligned[0]
                tokens = tuple(row[0].lower() for row in sentences[si][lo:hi + 1])
                item.update(token_tuple_basis='full_passage_aligned_same_sentence', native_token_length=len(tokens),
                            native_sentence=si, native_word_start=lo, native_word_end_inclusive=hi)
            else:
                # A standalone fragment cache hit is weaker evidence than a candidate
                # at this coordinate; it is explicitly separated in every output.
                fragment = ssplit_token_pos_lemma(q['text'])
                tokens = tuple(line.split('\t')[0].lower() for line in fragment.splitlines() if line)
                flat = [(si, left, right) for si, sent in enumerate(locs) for left, right in sent]
                starts = {si for si, left, _ in flat if left == q['local_start']}
                ends = {si for si, _, right in flat if right == q['local_end']}
                item.update(token_tuple_basis='standalone_fragment_no_exact_passage_alignment', native_token_length=None,
                            crosses_sentence=bool(starts and ends and not starts & ends),
                            start_native_boundary=bool(starts), end_native_boundary=bool(ends))
            item['query_token_tuple'] = list(tokens)
            query_tuples[q['query_index']] = tokens
            matches = [(key, value) for key, value in expected.items()
                       if key[:2] == (q['local_start'], q['local_end'])]
            item['generated_coordinate_count'] = len(matches)
            assert all(value == tokens for _, value in matches)
    generator_seconds = time.monotonic() - generator_started

    # All bytes/row counts are checked. For replayed passages, every actual raw
    # coordinate and source word indices must equal the generator's exact set.
    raw_stats, raw_references = {}, defaultdict(list)
    for model, specs in plan['neural_raw'].items():
        begin = time.monotonic()
        raw_passages = defaultdict(Counter)
        n = 0
        for sp in specs:
            for row in checked_rows(sp):
                n += 1
                did, pi = str(row['pmc_id']), row['passage_index']
                assert did in docs
                passage = docs[did]['full_text'][pi]
                assert row['offset'] == passage['offset'] + row['local_start']
                assert row['length'] == row['local_end'] - row['local_start']
                assert passage['text'][row['local_start']:row['local_end']] == row['text']
                key = (row['local_start'], row['local_end'], row['sentence'], row['word_start'], row['word_end_inclusive'])
                if (did, pi) in generated:
                    raw_passages[did, pi][key] += 1
                qindices = span_queries.get((did, row['offset'], row['length']), [])
                if not qindices:
                    continue
                cells_checked(row['top3'])
                tokens = generated[did, pi][key]
                for qi in qindices:
                    assert tokens == query_tuples[qi]
                    info = dict(unique_index=row['unique_index'], raw_path=sp['path'], top3=row['top3'])
                    items[qi]['raw_rows'].setdefault(model, []).append(info)
                    raw_references[model, sp['path']].append((row['unique_index'], tokens, row['top3']))
        assert n == 296984
        for passage_key, expected in generated.items():
            assert raw_passages[passage_key] == Counter(expected.keys()), (model, passage_key)
        raw_stats[model] = dict(rows=n, replayed_passages=len(generated), replayed_candidates=generated_count,
                                seconds=time.monotonic() - begin)

    requested = set(query_tuples.values())
    unique_stats = {}
    for model, specs in plan['unique_raw'].items():
        begin = time.monotonic()
        retained = defaultdict(list)
        indices = {}
        n = 0
        for spec in specs:
            current = {}
            for row in checked_rows(spec):
                n += 1
                tokens = tuple(row['lower_tokens'])
                if tokens not in requested:
                    continue
                cells_checked(row['top3'])
                record = dict(path=spec['path'], unique_index=row['unique_index'], top3=row['top3'])
                retained[tokens].append(record)
                current[row['unique_index']] = (tokens, row['top3'])
            # Each coordinate raw uses its sibling unique cache index, not a
            # global union index. The inherited reuse is verified by equality.
            raw_path = str(Path(spec['path']).with_name('raw-neural-candidates.jsonl'))
            for index, tokens, top3 in raw_references[model, raw_path]:
                assert current[index] == (tokens, top3)
            indices[spec['path']] = len(current)
        for q in queries:
            items[q['query_index']]['tuple_cache'][model] = retained.get(query_tuples[q['query_index']], [])
        unique_stats[model] = dict(rows=n, requested_tuples_found=len(retained), seconds=time.monotonic() - begin)

    for model, specs in plan['canonical'].items():
        for spec in specs:
            for row in checked_rows(spec):
                for qi in span_queries.get((str(row['pmc_id']), row['offset'], row['length']), []):
                    items[qi]['canonical_rows'].setdefault(model, []).append(row['hpo_id'])
    # These are boundary-absent from a pool which unioned both canonical sources.
    assert all(not x['canonical_rows'] for x in items)

    from transformers import AutoTokenizer
    tokenizers = {model: AutoTokenizer.from_pretrained(spec['path'], use_fast=True,
                  do_lower_case=spec['do_lower_case'], local_files_only=True)
                  for model, spec in plan['tokenizers'].items()}
    encoded = {}
    for tokens in requested:
        for model, tokenizer in tokenizers.items():
            full = tokenizer(list(tokens), truncation=False, is_split_into_words=True)
            limited = tokenizer(list(tokens), max_length=32, truncation=True, is_split_into_words=True)
            encoded[model, tokens] = dict(untruncated_tokens=len(full['input_ids']),
                actual_input_tokens=len(limited['input_ids']), truncated=len(full['input_ids']) > 32,
                last_word_index=max((x for x in limited.word_ids() if x is not None), default=-1),
                input_ids_sha256=hashlib.sha256(json.dumps(limited['input_ids']).encode()).hexdigest())
    assert 'tensorflow' not in sys.modules and 'torch' not in sys.modules
    for q, item in zip(queries, items):
        present = {model: bool(item['raw_rows'].get(model)) for model in plan['neural_raw']}
        assert len(set(present.values())) == 1 and next(iter(present.values())) == bool(item['generated_coordinate_count'])
        cache_presence = {model: bool(item['tuple_cache'].get(model)) for model in plan['neural_raw']}
        assert len(set(cache_presence.values())) == 1
        item['category'] = ('generated_coordinate_native_removed' if item['generated_coordinate_count'] else
                            'cached_tuple_only_no_coordinate' if any(cache_presence.values()) else 'neither_coordinate_nor_tuple')
        nt = item['native_token_length']
        item['boundary_group'] = ('native_unigram' if nt == 1 else 'aligned_over8' if nt and nt > 8 else
                                  'aligned_2_to_8' if nt else 'cross_sentence' if item.get('crosses_sentence') else 'unaligned_boundary')
        item['tokenizers'] = {m: encoded[m, query_tuples[q['query_index']]] for m in tokenizers}
        item['head_literal_present'] = q['gold_literal_id'] in labels
        item['head_current_reachable'] = q['gold_literal_id'] in set(cmap.values())
        item['conditional_support'] = {}
        item['raw_top1_gate'] = {}
        for k in (1, 3):
            supplies = {}
            for model in plan['neural_raw']:
                supplies[model] = {cmap[c['hpo_id']] for row in item['raw_rows'].get(model, []) for c in row['top3'][:k]
                                   if cmap.get(c['hpo_id'])}
                item['conditional_support'][f'{model}_top{k}'] = correct(supplies[model], q)
            item['conditional_support'][f'union_top{k}'] = correct(set.union(*supplies.values()), q)
        for model in plan['neural_raw']:
            rows = item['raw_rows'].get(model, [])
            item['raw_top1_gate'][model] = dict(
                valid_and_rounded_gt_95=any(cmap.get(row['top3'][0]['hpo_id']) and float(row['top3'][0]['native_probability_5dp']) > .95 for row in rows),
                literal_correct_and_rounded_gt_95=any(correct({cmap[row['top3'][0]['hpo_id']]}, q) and float(row['top3'][0]['native_probability_5dp']) > .95
                    for row in rows if cmap.get(row['top3'][0]['hpo_id'])))

    def aggregate(selected):
        chosen = [items[i] for i in selected]
        spans = {(queries[i]['pmc_id'], queries[i]['offset'], queries[i]['length']) for i in selected}
        return dict(atoms=len(chosen), unique_spans=len(spans),
                    linked=sum(x['gold_kind'] == 'linked' for x in chosen), unlinked=sum(x['gold_kind'] == 'unlinked' for x in chosen),
                    categories=dict(Counter(x['category'] for x in chosen)),
                    boundary_groups=dict(Counter(x['boundary_group'] for x in chosen)),
                    standalone_fragment_tuple_hits=sum(x['category'] == 'cached_tuple_only_no_coordinate' and
                        x['token_tuple_basis'].startswith('standalone') for x in chosen),
                    multi_gold_atoms=sum(x['gold_same_span_atomic_count'] > 1 for x in chosen),
                    conditional_support={key: sum(x['conditional_support'][key] for x in chosen)
                        for key in items[0]['conditional_support']},
                    truncation={model: sum(x['tokenizers'][model]['truncated'] for x in chosen) for model in tokenizers},
                    raw_top1_gate={model: {key: sum(x['raw_top1_gate'][model][key] for x in chosen)
                        for key in ('valid_and_rounded_gt_95', 'literal_correct_and_rounded_gt_95')} for model in tokenizers})

    result = dict(status='complete diagnostic only; no formal candidates or predictions',
        completed_at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__), plan_sha256=sha(args.plan),
        queries_sha256=sha(plan['queries']['path']), all=aggregate(range(len(items))),
        by_category={c: aggregate([i for i, x in enumerate(items) if x['category'] == c]) for c in sorted({x['category'] for x in items})},
        by_mechanism={c: aggregate([i for i, x in enumerate(items) if x['mechanism'] == c]) for c in sorted({x['mechanism'] for x in items})},
        by_gold_kind={c: aggregate([i for i, x in enumerate(items) if x['gold_kind'] == c]) for c in ('linked', 'unlinked')},
        generator=dict(passages=len(qp), tokens=verified_tokens, candidates=generated_count,
                       both_models_raw_coordinates_exact=True, seconds=generator_seconds),
        raw_scans=raw_stats, unique_scans=unique_stats, canonical_query_spans_present=0,
        matching_fixtures=dict(nil_cases=3, literal_not_canonical_case=1),
        no_new_NN=True, no_weights_loaded=True, originaldev_read=False, new_fit=False,
        seconds=time.monotonic() - started, limitations=plan['limitations'])
    assert result['seconds'] < 120
    audit = out / 'per-query-audit.json'
    audit.write_text(json.dumps(items, ensure_ascii=False, indent=2) + '\n')
    result['audit'] = dict(path=str(audit), sha256=sha(audit), bytes=audit.stat().st_size)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
