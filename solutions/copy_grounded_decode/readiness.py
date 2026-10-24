"""No-model, no-gold, no-score supply and native-token contract preparation."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

WORKSPACE = Path(os.environ.get('PATIENTPHEX_WORKSPACE', '/workspace'))
GROUNDED_SOURCE = Path(os.environ.get('GROUNDED_SOURCE_ROOT', str(WORKSPACE/'patientphex-worktrees/explore-grounded-generation')))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.append(str(GROUNDED_SOURCE))
from solutions.copy_grounded_decode.triple_trie import TripleList, compact
from solutions.grounded_generation.schema import locate, grounded, units
from solutions.grounded_generation.inference.decode import interpret

R = WORKSPACE/'patientphex-runs/round2-20261023/research'
FILES = {
    'prompts': (R/'grounded-generation-v1/targets-v2/F1.input.jsonl', 'c6426db4c5183c137c7798fc7fe1abe326646ed08391aa240a2cd304ae8dcfc0'),
    'pool': (R/'mention-canonical-normalization-v1/prepared-v1/F1.input.jsonl', '6ff12825202539eb1a309300c3b17030ee08ba450c5cf9bf7de0176baa7e6621'),
    'input': (WORKSPACE/'patientphex-data/round2-inner4/fold-1/validation16.input.jsonl', '962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'),
    'raw': (R/'grounded-generation-v1/infer-F1-v1/terminal/output/responses.jsonl', 'db5154ae89a6d30ababd56666f252fac5e91936f8f4f7e047abd36a2e145c244'),
    'normalizations': (R/'grounded-generation-v1/F1-normalized-v1/normalizations.jsonl', '786cc560a9678d989703d062c65dc88c6f1ef6adc2d4e15c8a061c42dbb15dec'),
    'pure_G': (R/'grounded-generation-v1/quality-F1-metadata-recovery-v1/assembled/new-only.jsonl', '80bc3dc728bb7f93bfeda726b66596ec293160d4ac1333c2e7f9a31911250e65'),
    'B4': (R/'four-source-fusion-20261024/F1-v1/prediction.jsonl', '5ad5cfe33ebea607c0012ee04420b31ff54cad06a52305091c54bb7d7ed7207e'),
    'S_with_B': (R/'nonthinking-sampled-consensus-native-20261024/assembled-v1/primary-B-union-S1-intersect-S2.pred.jsonl', '31d090f0cd0bfec43b61243a68e425eac224e785df32a2f1be76ce4d6ff6497e'),
    'S_inherited_B': (R/'public-new-lgb-support-control-20261024/actual-v1/fold1/supported.pred.jsonl', '2a65056f8c47738b8a5d8f54a8e36bbc68e68e9a3657a237c56ccd9f3241b7d0'),
    'Q': (R.parent/'stage8-qwen3-relations/quality-v1/single.prediction.jsonl', '103cab0f1eae88356bc164ed5b8924272642c686a3e031652010e584f2d7a9c4'),
    'C': (R/'patient-response-code-v1/F1-full80-v1/quality/coded-alone/pred.jsonl', 'a5a75d050029038703641e8242c0d6b9557a0e4ac8c69914b0d8859e5857517b'),
    'pool_strong_B': (R/'lightgbm-selector-readiness/formal-control/fold1/lightgbm.nested.jsonl', '8cb61ba94fab8de4223acafeece83ef5a243954312d008073a42e5ba80398d6f'),
    'pool_gliner': (R/'gliner25-task1-v1/normalization-v1-collected/actual/entities.json', '6ac9aa51a14b0cfd5f289f15dc4e3e29e8e375401e8e872f9a36a4a13f46e82d'),
    'tokenizer': (R.parent/'case_retrieval/qwen35-preparation/verified-small/tokenizer.json', '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'),
}


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def rows(name):
    return [json.loads(x) for x in FILES[name][0].read_text().splitlines()]


def relations(records):
    assert len(records) == len({str(d['pmc_id']) for d in records}) == 16
    assert sum(len(d['association']) for d in records) == 39
    assert all(len(d['association']) == len({a['patient_id'] for a in d['association']}) for d in records)
    return {(str(d['pmc_id']), a['patient_id'], cid) for d in records
            for a in d['association'] for cid in a['phenotype']}


def expect_reject(fn):
    try:
        fn()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Expected strict rejection')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    start = time.monotonic()
    allowed_json = {p.resolve() for p, _ in FILES.values()}

    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            name = Path(arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0])
            if '.gold.' in str(name):
                raise PermissionError('Held gold forbidden')
            mode = arguments[1]
            if isinstance(mode, str) and mode.startswith('r') and name.suffix in ('.json', '.jsonl'):
                if name.resolve() not in allowed_json:
                    raise PermissionError('Data read outside frozen allowlist: ' + str(name))
    sys.addaudithook(audit)
    import tokenizers
    from tokenizers import Tokenizer
    inputs = {name: ref(p) for name, (p, expected) in FILES.items()}
    assert all(inputs[k]['sha256'] == expected for k, (_, expected) in FILES.items())
    inherited_sources = {}
    for name, expected in [('schema.py','460dd7cce3208a7aca521f73e6a675959eb649b562312a877eeac0f2af36edbf'),
                           ('inference/decode.py','8e98f9ef2c1e123e9d071dc83f525a92ba90f26fffcca89532bcd78ac2b61305')]:
        inherited_sources[name] = ref(GROUNDED_SOURCE/'solutions/grounded_generation'/name)
        assert inherited_sources[name]['sha256'] == expected
    tokenizer = Tokenizer.from_file(str(FILES['tokenizer'][0]))
    docs = {str(x['pmc_id']): x for x in rows('input')}
    prompts = rows('prompts')
    pool_rows = rows('pool')
    assert len(docs) == 16 and len(prompts) == 39 and len(pool_rows) == 1416
    assert all(set(x) == {'pmc_id', 'pmid', 'patient', 'full_text'} for x in docs.values())
    for row in prompts:
        assert set(row) == {'row_id', 'pmc_id', 'target_patient_id', 'messages', 'units'}
        assert row['units'] == units(docs[str(row['pmc_id'])])
    pools = defaultdict(set)
    for row in pool_rows:
        assert set(row) == {'row_id', 'pmc_id', 'quote', 'offset', 'length', 'messages'}
        did = str(row['pmc_id'])
        triple = locate(docs[did], row['quote'], row['offset'])
        assert triple is not None and row['length'] == len(row['quote'])
        anchor = grounded(triple, {u['unit_id']: u for u in units(docs[did])})
        assert anchor['offset'] == row['offset'] and anchor['length'] == row['length']
        pools[did].add(tuple(triple))
    assert sum(map(len, pools.values())) == 1416
    quotes = {did: {x[1] for x in values} for did, values in pools.items()}
    raw = rows('raw')
    assert len(raw) == 39 and {x['row_id'] for x in raw} == {x['row_id'] for x in prompts}
    original = Counter()
    for record in raw:
        decision = interpret(record['response'], docs[str(record['pmc_id'])], record['actual_last_token_eos'], record['hit_token_limit'])
        original['rows'] += 1
        original['EOS'] += record['actual_last_token_eos']
        original['capped'] += record['hit_token_limit']
        original['items'] += decision['generated_items']
        original['valid_anchors'] += len(decision['anchors'])
        original['rejected'] += len(decision['rejected'])
        for item in decision['rejected']:
            original['rejected:' + item['reason']] += 1
        did = str(record['pmc_id'])
        for anchor in decision['anchors']:
            key = (anchor['unit_id'], anchor['quote'], anchor['occurrence_index'])
            original['valid_anchor_exact_triple_in_pool'] += key in pools[did]
            original['valid_anchor_quote_anywhere_in_doc_pool'] += key[1] in quotes[did]
    pure = relations(rows('pure_G'))
    baseline = relations(rows('B4'))
    inherited_B = relations(rows('S_inherited_B'))
    assert inherited_B <= baseline
    outside = pure - baseline
    support_sets = {name:relations(rows(name)) for name in ('S_with_B','Q','C')}
    support_union = pure.union(*support_sets.values())
    broad_outside = support_union - baseline
    # Source proposal IDs are a no-gold diagnostic only: never prompt/grammar IDs.
    pool_codes = defaultdict(set)
    proposer_spans = set()
    for doc in rows('pool_strong_B'):
        pool_codes[str(doc['pmc_id'])].update(e['identifier'] for e in doc['entities'])
        proposer_spans.update((str(doc['pmc_id']), e['offset'], e['length'], e['text']) for e in doc['entities'])
    for did, entities in json.loads(FILES['pool_gliner'][0].read_text()).items():
        pool_codes[str(did)].update(e['identifier'] for e in entities)
        proposer_spans.update((str(did), e['offset'], e['length'], e['text']) for e in entities)
    assert proposer_spans == {(str(r['pmc_id']),r['offset'],r['length'],r['quote']) for r in pool_rows}
    source_id_covered = {r for r in broad_outside if r[2] in pool_codes[r[0]]}
    exact_supported = set()
    quote_supported = set()
    for item in rows('normalizations'):
        if item['status'] != 'mapped':
            continue
        did = str(item['pmc_id'])
        key = (item['unit_id'], item['quote'], item['occurrence_index'])
        rels = {(did, item['target_patient_id'], cid) for cid in item['ids']}
        if key in pools[did]:
            exact_supported.update(rels)
        if key[1] in quotes[did]:
            quote_supported.update(rels)
    supply = dict(pure_G=len(pure), B4=len(baseline), pure_G_outside_B4=len(outside),
                  old_G_exact_anchor_witnesses=len(pure & exact_supported),
                  old_G_same_quote_witnesses=len(pure & quote_supported),
                  outside_B4_exact_anchor_witnesses=len(outside & exact_supported),
                  outside_B4_same_quote_witnesses=len(outside & quote_supported),
                  old_sources_outside_B4={n:len(v-baseline) for n,v in support_sets.items()},
                  broad_union_outside_B4=len(broad_outside),
                  broad_only_outside_B4=len(broad_outside-outside),
                  broad_outside_pool_proposed_ID_overlap=len(source_id_covered),
                  G_only_pool_proposed_ID_overlap=len(source_id_covered & outside),
                  broad_only_pool_proposed_ID_overlap=len(source_id_covered-outside),
                  S_inherited_B_fully_contained_in_B4=len(inherited_B),
                  interpretation='Prediction overlap only, not TP/recall. Old-G quote witnesses and pool-proposer IDs are not exact future normalization bounds. Support union count is the fixed primary additions ceiling.')
    contracts = Counter()
    pool_detail = []
    grammar_start = time.monotonic()

    def check_sequence(grammar, triples, callback=True):
        ids = grammar.serialize_ids(triples)
        text = compact(triples)
        assert tokenizer.decode(ids, skip_special_tokens=False) == text
        assert grammar.strict_parse(text, eos=True, capped=False) == [list(x) for x in triples]
        if callback:
            for i, token in enumerate(ids):
                assert token in grammar.allowed(ids[:i])
                assert token in grammar.callback(2)(0, [17, 19] + ids[:i])
            assert grammar.allowed(ids) == [248044, 248046]
            for eos in [248044, 248046]:
                assert grammar.allowed(ids + [eos]) == [248044, 248046]
            contracts['callback_tokens'] += len(ids)
        contracts['complete_strings'] += 1
        return ids

    for did, triples in sorted(pools.items()):
        ordered = sorted(triples)
        grammar = TripleList(tokenizer, ordered, [248044, 248046])
        check_sequence(grammar, [])
        for triple in ordered:
            check_sequence(grammar, [triple])
        # Every actual atom participates in multi-item prefixes, in both orders.
        for order in (ordered, list(reversed(ordered))):
            check_sequence(grammar, order, callback=False)
            for i in range(0, len(order), 8):
                check_sequence(grammar, order[i:i + 8])
            text = compact(order)
            starts = {a for _, (a, b) in tokenizer.pre_tokenizer.pre_tokenize_str(text)}
            cursor = 3
            for triple in order:
                assert cursor in starts and text[cursor] == 'U'
                cursor += len(compact(triple)[2:-1]) + 4
                contracts['native_pretoken_block_boundaries'] += 1
        expect_reject(lambda: grammar.strict_parse(compact([ordered[0]] * 2), eos=True, capped=False))
        expect_reject(lambda: grammar.strict_parse(compact([ordered[0]]), eos=False, capped=False))
        expect_reject(lambda: grammar.strict_parse(compact([ordered[0]]), eos=True, capped=True))
        expect_reject(lambda: grammar.allowed([999999]))
        duplicate_ids = grammar.encode(compact([ordered[0]] * 2))
        assert any(token not in grammar.allowed(duplicate_ids[:i]) for i, token in enumerate(duplicate_ids))
        contracts['rejection_cases'] += 5
        pool_detail.append(dict(pmc_id=did, exact_triples=len(ordered), unique_quotes=len(quotes[did]),
                                trie_nodes=grammar.nodes, singleton_max_tokens=max(len(grammar.serialize_ids([x])) for x in ordered)))
    empty = TripleList(tokenizer, [], [248044, 248046])
    check_sequence(empty, [])
    assert empty.allowed([]) == empty.empty[:1]
    synthetic = [('U0000', 'café "quote"\\line\n雪', 0), ('U0000', 'café "quote"\\line\n雪', 1), ('U0099', 'x', 10)]
    sg = TripleList(tokenizer, synthetic, [248044, 248046])
    check_sequence(sg, synthetic)
    check_sequence(sg, list(reversed(synthetic)))
    expect_reject(lambda: sg.strict_parse('[["U0000","x",true]]', eos=True, capped=False))
    contracts['synthetic_unicode_escape_occurrence_and_empty_pool'] += 4
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    candidate_file = out/'candidates-by-patient.jsonl'
    candidate_file.write_text(''.join(json.dumps(dict(row_id=p['row_id'], pmc_id=p['pmc_id'], target_patient_id=p['target_patient_id'], triples=sorted(pools[str(p['pmc_id'])])), ensure_ascii=False) + '\n' for p in prompts))
    allowed_json.add(candidate_file.resolve())
    result = dict(status='PASS_CPU_NATIVE_FINITE_TRIPLE_READINESS_NOT_GENERATION', inputs=inputs,
                  sources={n:ref(Path(__file__).parent/n) for n in ('readiness.py', 'triple_trie.py')},
                  inherited_sources=inherited_sources,
                  tokenizers_version=tokenizers.__version__, pool_docs=len(pools), patients=len(prompts),
                  exact_triples=sum(map(len,pools.values())), doc_unique_quotes=sum(map(len,quotes.values())),
                  repeated_quote_extra_occurrences=sum(map(len,pools.values()))-sum(map(len,quotes.values())),
                  original_raw=dict(original), supply=supply, contracts=dict(contracts), pool_detail=pool_detail,
                  old_raw_generation_seconds=sum(r['seconds'] for r in raw),
                  old_raw_input_tokens=sum(r['input_tokens'] for r in raw),
                  old_raw_output_tokens=sum(r['output_tokens'] for r in raw),
                  old_raw_max_output_tokens=max(r['output_tokens'] for r in raw),
                  pool_artifact=ref(candidate_file), grammar_seconds=time.monotonic()-grammar_start,
                  total_seconds=time.monotonic()-start, model_calls=0, GPU_calls=0, fit_calls=0, quality_calls=0,
                  unchanged_prompt_file=True, held_gold_read=False,
                  limitations=['Actual HF callback/generate and future normalization not executed.',
                               'No accuracy claim; restrictive pool can remove useful free spans.',
                               'Raw IDs/native text must be persisted before strict parsing in future runtime.'])
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','sources','pool_detail')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
