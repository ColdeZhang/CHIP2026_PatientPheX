"""Read-only, label-free coverage of signals discarded by old three-bit stacking.

No fitter, quality script, model, remote client, or gold table is imported.
Only the frozen F1 input/three raw prediction sources are opened as data.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path


def ref(path):
    p = Path(path); b = p.read_bytes()
    return dict(path=str(p.resolve()), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-route', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    old = args.old_route
    spec = importlib.util.spec_from_file_location('old_literal_features', old / 'common.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert ref(old / 'common.py')['sha256'] == '37578e207d4f8be6f2faf880c99b7f0a1cf155a82075c806311f8f8c5c98a5a7'
    plan = json.loads((old / 'fit-plan.json').read_text())
    ancestry = json.loads((old / 'evidence/ancestry.json').read_text())
    assert ancestry['all_task_ancestors_exclude_F0'] and ancestry['no_B1_features_selection_or_cut']
    # candidates() uses only input and raw_sources. Never iterate/check all plan refs:
    # the old plan also contains gold references that this audit must not open.
    u, rows, source_counts = module.candidates(plan)
    assert len(u) == 16 and len(rows) == 561
    streams = {'S1': set(), 'S2': set()}
    for row in module.json_lines(plan['raw_sources']['sampled_diagnostics']):
        if row['stream'] in streams:
            streams[row['stream']].update(tuple(k) for k in row['relations'])
    patterns = defaultdict(list)
    counters = Counter(); docs = defaultdict(set)
    for row in rows:
        k = tuple(row['key']); bits = row['features']
        n = int(k in streams['S1']) + int(k in streams['S2'])
        assert bits[1] == int(n == 2)
        disagreement = int(n == 1)
        n_patients = len(u[k[0]])
        patterns[''.join(map(str, bits))].append((disagreement, n_patients))
        counters['sample_votes_' + str(n)] += 1
        if disagreement:
            docs['sample_disagreement'].add(k[0])
            counters['disagreement_majority_' + str(int(sum(bits) >= 2))] += 1
    summary = {k: dict(rows=len(v), sample_disagreement=sum(x[0] for x in v),
                      disagreement_values=sorted({x[0] for x in v}),
                      patient_count_values=sorted({x[1] for x in v}),
                      joint_context_values=len(set(v)))
               for k, v in sorted(patterns.items())}
    result = dict(status='COMPLETE_LABEL_FREE_F1_SIGNAL_COVERAGE_NOT_GAIN_EVIDENCE',
                  source=ref(__file__), old_source=ref(old/'common.py'),
                  old_plan=ref(old/'fit-plan.json'), inherited_ancestry=ref(old/'evidence/ancestry.json'),
                  opened_data=dict(input=plan['input'], **plan['raw_sources']),
                  counts=source_counts, discarded_signal_counts=dict(counters),
                  documents_with_disagreement=len(docs['sample_disagreement']),
                  patterns=summary,
                  old_patterns_with_new_disagreement_variation=sum(len(v['disagreement_values']) > 1 for v in summary.values()),
                  old_patterns_with_joint_context_variation=sum(v['joint_context_values'] > 1 for v in summary.values()),
                  gold_or_mixed_target_tables_opened=0, fit_calls=0, GPU_calls=0,
                  quality_calls=0, F0_raw_data_opened=0, original_dev_opened=0,
                  limitations=['Coverage is not evidence of predictive correctness or improvement.',
                               'F1 is proposed future meta-training, already used in earlier development; not a blind test.',
                               'No baseline entities, base selection, prior LR/tree scores, or 48-document proposer features.'])
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps({k: result[k] for k in ['status','counts','discarded_signal_counts','documents_with_disagreement','patterns']}))


if __name__ == '__main__':
    main()
