"""Small post-run aggregation of frozen diagnostic evidence, without raw rereads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    a = p.parse_args()
    started = time.monotonic()
    root = Path(a.root)
    qpath = root / 'diagnostic-queries.json'
    apath = root / 'terminal/output/per-query-audit.json'
    spath = root / 'terminal/output/summary.json'
    assert sha(qpath) == 'ba222b4c8792010a5a3cb3748b2be800b6b825f0c3459c31708828055203ef92'
    assert sha(apath) == 'e9def0737cae62eccc13a195041ab1b7c02afff7985188737224d621078ae897'
    assert sha(spath) == '60ca3b099a1a1ecf55ef5eb44e86820b9a2deb7a7f04011f1df488f99f767e06'
    queries = json.loads(qpath.read_text())['queries']
    items = json.loads(apath.read_text())
    assert len(items) == len(queries) == 384
    span = lambda i: (queries[i]['pmc_id'], queries[i]['offset'], queries[i]['length'])

    def count(indices):
        selected = [items[i] for i in indices]
        return dict(atoms=len(indices), unique_spans=len({span(i) for i in indices}),
                    linked=sum(x['gold_kind'] == 'linked' for x in selected),
                    unlinked=sum(x['gold_kind'] == 'unlinked' for x in selected))

    groups = {}
    for name in ('generated_coordinate_native_removed', 'cached_tuple_only_no_coordinate', 'neither_coordinate_nor_tuple'):
        indices = [i for i, x in enumerate(items) if x['category'] == name]
        groups[name] = count(indices)
        groups[name]['by_boundary_group'] = {g: count([i for i in indices if items[i]['boundary_group'] == g])
                                           for g in sorted({items[i]['boundary_group'] for i in indices})}
        groups[name]['native_word_lengths_atoms'] = dict(sorted(Counter(
            str(items[i]['native_token_length']) for i in indices).items()))
        groups[name]['linked_head_current_unreachable'] = sum(items[i]['gold_kind'] == 'linked' and
            not items[i]['head_current_reachable'] for i in indices)
    generated = [i for i, x in enumerate(items) if x['generated_coordinate_count']]
    gates = {}
    for key in ('valid_and_rounded_gt_95', 'literal_correct_and_rounded_gt_95'):
        accepted = [i for i in generated if any(items[i]['raw_top1_gate'][m][key] for m in ('pubmed', 'bioformer'))]
        gates[key] = count(accepted)
    byfold = []
    for fold in range(4):
        indices = [i for i, q in enumerate(queries) if q['fold'] == fold]
        byfold.append(dict(fold=fold, **count(indices), generated=sum(items[i]['generated_coordinate_count'] > 0 for i in indices),
                           union_top1=sum(items[i]['conditional_support']['union_top1'] for i in indices),
                           union_top3=sum(items[i]['conditional_support']['union_top3'] for i in indices)))
    result = dict(status='post-run aggregation only, no new method or prediction', source_sha256=sha(__file__),
                  inputs={str(p): sha(p) for p in (qpath, apath, spath)}, categories=groups,
                  raw_top1_either_model_passes_original_gate_but_canonical_absent=gates, folds=byfold,
                  multi_gold=count([i for i, x in enumerate(items) if x['gold_same_span_atomic_count'] > 1]),
                  seconds=time.monotonic() - started,
                  interpretation=[
                      'The three categories refer to the neural ngram supply; dictionary resources and the generator are unchanged.',
                      'Gate-passing raw spans absent from final canonical show later native loss, but hybrid/abbr causes are not individually localized.',
                      'The 167 absent atoms include repeated atomic IDs at compound spans; use unique-span counts when discussing geometry.',
                      'The three >32 token fragments were not generated neural inputs; their tokenizer truncation is hypothetical, not an actual cached inference truncation.',
                      'Top3 literal coverage is conditional supply only, with no precision or one-span output attainability guarantee.'
                  ])
    destination = root / 'aggregate-supplement.json'
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
