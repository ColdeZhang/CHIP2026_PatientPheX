"""Input-only native unigram supply census. No labels, weights or NN calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
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
    h, n = hashlib.sha256(), 0
    with Path(spec['path']).open('rb') as stream:
        for line in stream:
            h.update(line)
            n += 1
            yield json.loads(line)
    assert h.hexdigest() == spec['sha256']
    assert Path(spec['path']).stat().st_size == spec['bytes'] and n == spec['rows']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    a = parser.parse_args()
    start = time.monotonic()
    signal.alarm(115)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert all(os.environ.get(k) == '0' for k in ('USE_TORCH', 'USE_TF', 'USE_FLAX'))
    plan = json.loads(Path(a.plan).read_text())
    assert sha(__file__) == plan['source_sha256']
    out = Path(plan['output_directory'])
    out.mkdir(exist_ok=False)
    root = Path(plan['public_resource_root'])
    for rel, digest in plan['resources'].items():
        assert sha(root / rel) == digest, rel
    for spec in plan['extra_resources']:
        assert sha(spec['path']) == spec['sha256'], spec['path']
    docs = list(map(json.loads, (root / 'train64-cache/train64.input.jsonl').read_text().splitlines()))
    assert len(docs) == 64 and len({str(d['pmc_id']) for d in docs}) == 64
    assert all(set(d) == {'pmc_id', 'pmid', 'patient', 'full_text'} for d in docs)
    sys.path.insert(0, str(root / 'native-pipeline/src'))
    import nltk
    nltk.data.path = [str(root / 'nltk_data')]
    from ssplit_tokenzier import ssplit_token_pos_lemma
    from restore_index import restore_index_nest_fn
    from dic_ner import dic_ont
    import ml_ner
    fixed_path = root / 'native-pipeline/fixed_candidate_function.py'
    fixed = fixed_path.read_text()
    assert fixed.count('range(2,sentence_ngram+1)') == 1
    unigram_source = fixed.replace('range(2,sentence_ngram+1)', 'range(1,2)')
    namespace = dict(ml_ner.__dict__)
    exec(compile(unigram_source, 'unigram_branch_only', 'exec'), namespace)
    build = namespace['build_ngram_testset_filted']
    (out / 'executed_unigram_function.py').write_text(unigram_source)
    dic = dic_ont({'dic_file': str(root / 'native-pipeline/dict/noabb_lemma.dic'),
                  'word_hpo_file': str(root / 'native-pipeline/dict/word_id_map.json'),
                  'hpo_word_file': str(root / 'native-pipeline/dict/id_word_map.json')})
    counts = Counter()
    filtered_pos, accepted_pos = Counter(), Counter()
    words, perdoc, rows = Counter(), [], []
    all_spans, accepted_spans, dictionary_spans = set(), set(), set()
    token_details = {}
    for doc in docs:
        did = str(doc['pmc_id'])
        dc = Counter()
        for pi, passage in enumerate(doc['full_text']):
            text = passage['text']
            if not text.strip():
                counts['empty_passages'] += 1
                continue
            conll = ssplit_token_pos_lemma(text)
            if not conll.strip():
                counts['empty_tokenized_passages'] += 1
                continue
            sentences = [[line.split('\t') for line in part.splitlines()] for part in conll.strip().split('\n\n')]
            dc['passages'] += 1
            dc['sentences'] += len(sentences)
            annotation = ''
            for sentence in sentences:
                toks = [r[0] for r in sentence]
                annotation += ' '.join(toks) + '\n' + ''.join(f'{j}\t{j}\t{w}\tHP:None\t0.0\n' for j, w in enumerate(toks)) + '\n'
            locations = restore_index_nest_fn(text, annotation)
            assert len(locations) == sum(map(len, sentences))
            normalized = text.replace('``', '" ').replace("''", '" ').lower()
            locs, at, expected = [], 0, []
            for si, sentence in enumerate(sentences):
                sl = [(int(r[0]), int(r[1])) for r in locations[at:at+len(sentence)]]
                for wi, (row, (left, right)) in enumerate(zip(sentence, sl)):
                    token, pos = row[0], row[3]
                    assert 0 <= left < right <= len(text)
                    assert normalized[left:right] == token.lower().replace('``', '"').replace("''", '"')
                    key = (did, passage['offset'] + left, right-left)
                    assert key not in all_spans
                    all_spans.add(key)
                    dc['tokens'] += 1
                    pun, tag = bool(ml_ner.pun_filter([token])), bool(ml_ner.pos_filter([pos], [token]))
                    dc['pun_filter_reject'] += int(pun)
                    dc['pos_filter_reject'] += int(tag)
                    dc['both_filters_reject'] += int(pun and tag)
                    if pun or tag:
                        filtered_pos[pos] += 1
                        continue
                    expected.append([si, wi, wi])
                    accepted_spans.add(key)
                    dc['accepted_coordinates'] += 1
                    accepted_pos[pos] += 1
                    word = token.lower()
                    words[word] += 1
                    token_details.setdefault(word, dict(native_cases=set(), POS=set()))
                    token_details[word]['native_cases'].add(token)
                    token_details[word]['POS'].add(pos)
                    rows.append(dict(pmc_id=did, passage_index=pi, sentence=si, word_index=wi,
                        offset=key[1], length=key[2], local_start=left, local_end=right,
                        text=text[left:right], native_token=token, lower_tokens=[word], POS=pos))
                locs.append(sl)
                at += len(sentence)
            ml_input, _, indices = build(conll)
            assert list(indices.values()) == expected
            instances = [x for x in ml_ner.ml_intext_fn(ml_input)[0] if x]
            assert len(instances) == len(expected)
            assert all(len(x) == 1 for x in instances)
            assert [x[0][0].lower() for x in instances] == [sentences[s][w][0].lower() for s, w, _ in expected]
            for x in restore_index_nest_fn(text, dic.matching(conll)):
                left, right = int(x[0]), int(x[1])
                dictionary_spans.add((did, passage['offset'] + left, right-left))
                dc['raw_dictionary_rows'] += 1
        counts.update(dc)
        perdoc.append(dict(pmc_id=did, **dc))
    generator_seconds = time.monotonic() - start
    assert counts['tokens'] == 205656
    assert counts['raw_dictionary_rows'] == 4130
    assert len(all_spans) == counts['tokens']
    assert len(accepted_spans) == len(rows) == counts['accepted_coordinates']
    assert counts['accepted_coordinates'] == counts['tokens'] - counts['pun_filter_reject'] - counts['pos_filter_reject'] + counts['both_filters_reject']
    requested = {(w,) for w in words}
    cache = {}
    for model, specs in plan['unique_raw'].items():
        t = time.monotonic()
        matched, total, length_dist = set(), 0, Counter()
        for spec in specs:
            for row in checked_rows(spec):
                key = tuple(row['lower_tokens'])
                total += 1
                length_dist[len(key)] += 1
                if key in requested:
                    matched.add(key)
        cache[model] = dict(scanned_rows=total, exact_unigram_tuple_reuse=len(matched),
            unigram_coordinates_with_reuse=sum(words[x[0]] for x in matched),
            new_unique_forward=len(requested-matched), token_tuple_length_histogram=dict(sorted(length_dist.items())),
            seconds=time.monotonic()-t, actual_binary_SHA_checked=True)
    from transformers import AutoTokenizer
    tokenizers = {}
    encoded = {w: {} for w in words}
    for model, spec in plan['tokenizers'].items():
        t = time.monotonic()
        tok = AutoTokenizer.from_pretrained(spec['path'], use_fast=True, do_lower_case=spec['do_lower_case'], local_files_only=True)
        lens, unk, truncated, ids_digest = Counter(), 0, 0, hashlib.sha256()
        for word in sorted(words):
            full = tok([word], truncation=False, is_split_into_words=True)
            limit = tok([word], max_length=32, truncation=True, is_split_into_words=True)
            n = len(full['input_ids'])
            lens[n] += 1
            unknown = tok.unk_token_id in full['input_ids']
            unk += int(unknown)
            truncated += int(n > 32)
            assert len(limit['input_ids']) <= 32
            assert all(x in (None, 0) for x in limit.word_ids())
            assert all(0 <= i < tok.vocab_size for i in limit['input_ids'])
            encoded[word][model] = dict(full_tokens=n, truncated=n > 32, contains_UNK=unknown,
                input_ids=limit['input_ids'], token_type_ids=limit['token_type_ids'], attention_mask=limit['attention_mask'])
            ids_digest.update((json.dumps([word, encoded[word][model]], sort_keys=True) + '\n').encode())
        tokenizers[model] = dict(vocab_size=tok.vocab_size, do_lower_case=tok.do_lower_case,
            native_word_lowercasing_before_tokenizer=True, max_length=32, padding='native post-padding to32',
            token_count_histogram=dict(sorted(lens.items())), truncated_unique=truncated, UNK_unique=unk,
            unique_inputs=len(words), input_record_sha256=ids_digest.hexdigest(), seconds=time.monotonic()-t)
    assert not any(k in sys.modules for k in ('tensorflow', 'torch'))
    canonical = {}
    for model, specs in plan['canonical'].items():
        spans = set()
        for spec in specs:
            for row in checked_rows(spec):
                spans.add((str(row['pmc_id']), row['offset'], row['length']))
        canonical[model] = dict(all_spans=len(spans), unigram_accepted_spans_present=len(spans & accepted_spans))
    for row in rows:
        key = (row['pmc_id'], row['offset'], row['length'])
        row['raw_dictionary_span_present'] = key in dictionary_spans
    with gzip.open(out / 'input-only-unigram-coordinates.jsonl.gz', 'wt', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    with gzip.open(out / 'input-only-unique-tokens.jsonl.gz', 'wt', encoding='utf-8') as f:
        for word in sorted(words):
            f.write(json.dumps(dict(lower_tokens=[word], count=words[word],
                native_cases=sorted(token_details[word]['native_cases']), POS=sorted(token_details[word]['POS']),
                models=encoded[word]), ensure_ascii=False, separators=(',', ':')) + '\n')
    result = dict(status='complete_input_only_unigram_supply_census_no_NN_no_quality',
        at_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__), plan_sha256=sha(a.plan),
        input_sha256=plan['resources']['train64-cache/train64.input.jsonl'], documents=64,
        counts=dict(counts), unique_lower_unigrams=len(words), accepted_POS=dict(accepted_pos), rejected_POS=dict(filtered_pos),
        dictionary=dict(raw_rows=counts['raw_dictionary_rows'], unique_spans=len(dictionary_spans),
            all_token_spans_present=len(dictionary_spans & all_spans), accepted_spans_present=len(dictionary_spans & accepted_spans),
            accepted_spans_not_in_raw_dictionary=len(accepted_spans - dictionary_spans),
            accepted_unique_words_with_any_dict_span=len({r['lower_tokens'][0] for r in rows if r['raw_dictionary_span_present']})),
        canonical=canonical, cache=cache, tokenizers=tokenizers, per_document=perdoc,
        function_delta='Only loop range(2,sentence_ngram+1) -> range(1,2); unchanged previously unreachable one-gram branch, pun_filter and pos_filter; old2-8 function and caches untouched.',
        upstream_fixed_function_sha256=sha(fixed_path), unigram_function_sha256=sha(out/'executed_unigram_function.py'),
        generator_and_dictionary_seconds=generator_seconds, total_seconds=time.monotonic()-start,
        cost_estimates={m: dict(unique_forward_needed=cache[m]['new_unique_forward'],
            prior_A100_unique_per_second=plan['prior_throughput'][m],
            prior_A100_rate_pure_forward_seconds=cache[m]['new_unique_forward']/plan['prior_throughput'][m],
            limitation='Estimate from old A100 batches; excludes model loading/transfer and is not another GPU measurement.') for m in cache},
        no_labels=True, no_originaldev=True, no_models_or_NN=True, no_new_quality=True,
        limitations=plan['limitations'])
    result['artifacts']={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()}
    assert result['total_seconds'] < 120
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','counts','unique_lower_unigrams','dictionary','cache','cost_estimates','total_seconds')}),flush=True)


if __name__ == '__main__':
    main()
