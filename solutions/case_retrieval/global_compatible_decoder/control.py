"""One fixed exact-compatible-set control over the frozen BioBERT probabilities."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import signal
import sys
import time

import census as c


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def field_span(line, wanted='entities'):
    """Locate only a top-level value; preserve all bytes outside that value."""
    decoder = json.JSONDecoder()
    pos = 0
    def whitespace(p):
        while p < len(line) and line[p].isspace():
            p += 1
        return p
    pos = whitespace(pos)
    assert line[pos] == '{'
    pos += 1
    names, found = set(), None
    while True:
        pos = whitespace(pos)
        if line[pos] == '}':
            break
        name, pos = decoder.raw_decode(line, pos)
        assert isinstance(name, str) and name not in names
        names.add(name)
        pos = whitespace(pos)
        assert line[pos] == ':'
        pos = whitespace(pos + 1)
        start = pos
        _, pos = decoder.raw_decode(line, pos)
        if name == wanted:
            found = start, pos
        pos = whitespace(pos)
        if line[pos] == '}':
            break
        assert line[pos] == ','
        pos += 1
    assert not line[pos+1:].strip() and found is not None
    return found


def replace_entities(line, entities):
    old = json.loads(line)
    if old['entities'] == entities:
        return line
    a, b = field_span(line)
    result = line[:a] + json.dumps(entities, ensure_ascii=False) + line[b:]
    x, y = field_span(result)
    assert (line[:a].encode(), line[b:].encode()) == (result[:x].encode(), result[y:].encode())
    assert {k:v for k,v in old.items() if k != 'entities'} == {k:v for k,v in json.loads(result).items() if k != 'entities'}
    return result


def solve_component(nodes, edges, weights, reference, ordering):
    """Exact objective; fixed old-overlap and lexicographic tie breaking."""
    assert all(weights[i] > 0 for i in nodes)
    best = None
    inspected = 0
    for bits in range(1 << len(nodes)):
        chosen = frozenset(nodes[j] for j in range(len(nodes)) if bits >> j & 1)
        if any(a in chosen and b in chosen for a, b in edges):
            continue
        inspected += 1
        utility = sum((weights[i] for i in chosen), Fraction(0))
        overlap = len(chosen & reference)
        lexical = tuple(sorted(ordering[i] for i in chosen))
        candidate = utility, overlap, lexical, chosen
        if best is None or candidate[:2] > best[:2] or (candidate[:2] == best[:2] and lexical < best[2]):
            best = candidate
    return best[3], dict(utility=str(best[0]), reference_overlap=best[1], feasible_subsets=inspected)


def universe(decoder, candidates, doc, allowed):
    anchors = decoder.greedy(candidates, .3)
    anchor_keys = {decoder.key(x) for x in anchors}
    current, _ = decoder.decode(candidates, anchors, doc, allowed, .3)
    eligible = []
    for index, candidate in enumerate(candidates):
        if candidate['selector_probability'] < .3:
            continue
        assert candidate['selector_probability'] > .3, 'Frozen census found no exact-cut item'
        if decoder.key(candidate) not in anchor_keys:
            ids = set(candidate['identifier'].split(';'))
            if not ids or not ids <= allowed:
                continue
            _, passage = decoder.prior.base.passage_for(doc, candidate['offset'])
            local = candidate['offset'] - int(passage.get('offset', 0))
            assert passage['text'][local:local+candidate['length']] == candidate['text']
            if decoder.prior.combine.ontology.is_negated(passage['text'], local, local+candidate['length']):
                continue
        eligible.append((index, candidate))
    return current, eligible


def optimize(decoder, candidates, doc, allowed):
    current, eligible = universe(decoder, candidates, doc, allowed)
    old_keys = {decoder.key(x) for x in current}
    original_indices = [x[0] for x in eligible]
    cs = [x[1] for x in eligible]
    weights = [Fraction.from_float(x['selector_probability']) - Fraction.from_float(.3) for x in cs]
    ordering = [(*decoder.key(x), i) for i, x in eligible]
    reference = {i for i, x in enumerate(cs) if decoder.key(x) in old_keys}
    assert len(reference) == len(current)
    edges = []
    for i, a in enumerate(cs):
        for j in range(i):
            b = cs[j]
            compatible = (decoder.prior.proper_containment(a, b) and
                          not set(a['identifier'].split(';')) & set(b['identifier'].split(';')))
            if decoder.prior.overlap(a, b) and not compatible:
                edges.append((j, i))
    assert all(not {a, b} <= reference for a, b in edges)
    groups = c.components(cs, edges)
    chosen, details = set(), []
    for group in groups:
        assert len(group) <= 3, 'Stop if frozen census contract changed'
        if len(group) == 1:
            assert group[0] in reference, 'Isolates must be retained unchanged'
            chosen.add(group[0])
            continue
        local_edges = [(a, b) for a, b in edges if a in group and b in group]
        selected, detail = solve_component(group, local_edges, weights, reference, ordering)
        old = reference.intersection(group)
        old_utility = sum((weights[i] for i in old), Fraction(0))
        assert Fraction(detail['utility']) >= old_utility
        details.append(dict(nodes=[dict(original_index=original_indices[i], key=decoder.key(cs[i]),
                                        probability=cs[i]['selector_probability'], exact_utility=str(weights[i])) for i in group],
                            edges=[[original_indices[a], original_indices[b]] for a, b in local_edges],
                            old_indices=sorted(original_indices[i] for i in old),
                            new_indices=sorted(original_indices[i] for i in selected),
                            old_utility=str(old_utility), **detail))
        chosen.update(selected)
    selected = sorted((decoder.entity(cs[i]) for i in chosen), key=lambda x:(x['offset'], x['length'], x['identifier']))
    assert all(not {a, b} <= chosen for a, b in edges)
    return current, selected, dict(eligible=len(cs), edges=len(edges), components=details,
                                  old_utility=str(sum((weights[i] for i in reference), Fraction(0))),
                                  new_utility=str(sum((weights[i] for i in chosen), Fraction(0))),
                                  added=len(chosen-reference), removed=len(reference-chosen))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    start = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('fixed 60-second CPU budget')))
    signal.alarm(60)
    plan = c.read(args.plan)
    assert plan['total_CPU_cap_seconds'] == 60
    for spec in plan['sources']:
        c.checked(spec)
    c.checked(plan['contracts'])
    old = c.read(c.checked(plan['old_plan']))
    old_freeze = c.read(c.checked(plan['old_freeze']))
    prepared = c.read(c.checked(old['prepared_manifest']))
    for path, digest in old['sources'].items():
        assert c.ref(path)['sha256'] == digest
    sys.path.insert(0, old['nested_source'])
    import decode_nested_selector as decoder
    _, allowed, _, _ = decoder.prior.base.parse_hpo(c.checked(old['hpo']))
    allowed.discard('HP:0000118')
    root = Path(plan['output_directory'])
    root.mkdir(exist_ok=False)
    predictions, diagnostics, frozen, seen = {}, [], [], set()
    for spec in old_freeze['folds']:
        f = spec['fold']
        scores = c.rows(c.checked(spec['scores']))
        inputs = c.rows(c.checked(prepared['folds'][f]['input']))
        old_bytes = c.checked(spec['new_prediction']).read_bytes()
        lines = old_bytes.decode('utf-8').splitlines(keepends=True)
        assert len(scores) == len(inputs) == len(lines) == 16
        output, new_docs, changed = [], [], 0
        for scored, row, line in zip(scores, inputs, lines):
            before = json.loads(line)
            assert before['pmc_id'] == scored['pmc_id'] == row['pmc_id'] and before['pmc_id'] not in seen
            seen.add(before['pmc_id'])
            assert set(scored) == {'pmc_id', 'candidates'}
            assert set(row['input']) == {'pmc_id','pmid','patient','full_text'}
            assert [{k:v for k,v in x.items() if k != 'selector_probability'} for x in scored['candidates']] == row['candidates']
            current, selected, detail = optimize(decoder, scored['candidates'], row['input'], allowed)
            assert before['entities'] == current
            after = replace_entities(line, selected)
            changed += current != selected
            output.append(after)
            new_docs.append(json.loads(after))
            diagnostics.append(dict(fold=f, pmc_id=before['pmc_id'], changed=current != selected, **detail))
        dest = root / f'fold{f}.jsonl'
        dest.write_bytes(''.join(output).encode('utf-8'))
        frozen.append(dict(fold=f, baseline=spec['new_prediction'], prediction=c.ref(dest),
                           changed_documents=changed, byteexact=dest.read_bytes() == old_bytes))
        predictions[f] = new_docs
    assert len(seen) == 64
    assert sum(d['eligible'] for d in diagnostics) == 4633
    assert sum(d['edges'] for d in diagnostics) == 70
    assert sum(len(d['components']) for d in diagnostics) == 66
    write(root/'components.json', diagnostics)
    changed = sum(d['changed'] for d in diagnostics)
    freeze = dict(frozen_at_utc=now(), plan=c.ref(args.plan), predictions=frozen,
                  components=c.ref(root/'components.json'), changed_documents=changed,
                  nonentity_literal_bytes_preserved=True, new_gold_read=False, new_quality_evaluated=False,
                  seconds_before_labels=time.monotonic()-start)
    write(root/'complete-freeze.json', freeze)
    for f in frozen:
        c.checked(f['prediction'])
    baseline = [c.read(c.checked(x)) for x in plan['baseline_fold_metrics']]
    old_pooled = c.read(c.checked(plan['baseline_pooled_metrics']))
    if changed == 0:
        assert all(x['byteexact'] for x in frozen)
        reports, pooled = baseline, old_pooled
        gold_read = False
    else:
        write(root/'evaluation-start.json', dict(started_utc=now(), freeze=c.ref(root/'complete-freeze.json')))
        sys.path.insert(0, old['evaluation_source'])
        import evaluate
        import validate_predictions as validator
        assert evaluate.VERSION == 'patientphex-public-approx-v2'
        assert c.ref(evaluate.__file__)['sha256'] == old['sources'][str(Path(evaluate.__file__).resolve())]
        assert c.ref(validator.__file__)['sha256'] == old['sources'][str(Path(validator.__file__).resolve())]
        gold = c.rows(c.checked(old['train64_gold']))
        truth = {x['pmc_id']:x for x in gold}
        assert len(truth) == len(gold) == 64 and set(truth) == seen
        hpo = validator.read_hpo(c.checked(old['hpo']))
        reports, byid = [], {}
        for f in range(4):
            docs = predictions[f]
            labels = [truth[x['pmc_id']] for x in docs]
            report = evaluate.evaluate_report(labels, docs, 'all', 'text')
            valid = validator.validate_predictions(labels, docs, hpo)
            assert valid['valid'] and valid['error_count'] == valid['warning_count'] == 0
            assert report['counts']['relation'] == baseline[f]['counts']['relation']
            reports.append(report)
            write(root/f'fold{f}.metrics.json', report)
            write(root/f'fold{f}.validation.json', valid)
            byid.update({x['pmc_id']:x for x in docs})
        pooled = evaluate.evaluate_report(gold, [byid[x['pmc_id']] for x in gold], 'all', 'text')
        assert pooled['counts']['relation'] == old_pooled['counts']['relation']
        write(root/'pooled.metrics.json', pooled)
        gold_read = True
    deltas = [reports[f]['metrics']['Score'] - baseline[f]['metrics']['Score'] for f in range(4)]
    pooled_delta = pooled['metrics']['Score'] - old_pooled['metrics']['Score']
    result = dict(status='complete_fixed_control', finished_utc=now(), seconds=time.monotonic()-start,
                  source=c.ref(__file__), plan=c.ref(args.plan), freeze=c.ref(root/'complete-freeze.json'),
                  changed_documents=changed, all_four_byteexact=all(x['byteexact'] for x in frozen),
                  added=sum(d['added'] for d in diagnostics), removed=sum(d['removed'] for d in diagnostics),
                  changed_components=sum(x['new_indices'] != x['old_indices'] for d in diagnostics for x in d['components']),
                  total_exact_utility_gain=str(sum((Fraction(d['new_utility'])-Fraction(d['old_utility']) for d in diagnostics), Fraction(0))),
                  fold_baseline_metrics=[x['metrics'] for x in baseline], fold_new_metrics=[x['metrics'] for x in reports],
                  fold_score_deltas=deltas, pooled_baseline=old_pooled['metrics'], pooled_new=pooled['metrics'],
                  pooled_score_delta=pooled_delta, keep=pooled_delta > 0 and all(d >= 0 for d in deltas),
                  gold_read=gold_read, new_evaluator_calls=5 if gold_read else 0,
                  baseline_metrics_reused=True, gpu_used=False, model_fit=False, threshold_search=False,
                  originaldev_read=False, official_actions=False)
    assert result['seconds'] < 60
    write(root/'result.json', result)
    signal.alarm(0)
    print(json.dumps({k:result[k] for k in ('changed_documents','changed_components','all_four_byteexact','added','removed','total_exact_utility_gain','pooled_score_delta','keep','seconds','new_evaluator_calls')}))


if __name__ == '__main__':
    main()
