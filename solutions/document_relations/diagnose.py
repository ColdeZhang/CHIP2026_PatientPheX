"""Post-endpoint counts and ranking diagnostics; never fit or select a cutoff."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from evaluate import read_jsonl, relations, write


def ranking(rows):
    groups = defaultdict(lambda: [0, 0])
    for score, truth in rows:
        groups[score][int(truth)] += 1
    positives = sum(v[1] for v in groups.values())
    negatives = sum(v[0] for v in groups.values())
    seen_negatives = concordant = 0
    for score in sorted(groups):
        no, yes = groups[score]
        concordant += yes * (seen_negatives + no / 2)
        seen_negatives += no
    seen_positive = seen = ap = 0
    for score in sorted(groups, reverse=True):
        no, yes = groups[score]
        seen_positive += yes
        seen += no + yes
        ap += yes * seen_positive / seen
    return {'pairs': len(rows), 'positives': positives, 'negatives': negatives,
            'auroc': concordant / (positives * negatives) if positives and negatives else None,
            'average_precision': ap / positives if positives else None}


def main():
    ap = argparse.ArgumentParser()
    for name in ['scores', 'held_encoded', 'gold', 'baseline', 'quality', 'output']:
        ap.add_argument('--' + name.replace('_', '-'), required=True)
    args = ap.parse_args()
    quality = json.loads((Path(args.quality) / 'result.json').read_text())
    assert quality['new_v2_calls'] == 2 and quality['old_v2_calls'] == 0
    scores = read_jsonl(args.scores)
    encoded = json.load(gzip.open(args.held_encoded, 'rt'))
    assert len(encoded) == 16 and len(scores) == 2874
    expected = {tuple(key) for doc in encoded for key in doc['keys']}
    assert {tuple(row['key']) for row in scores} == expected
    assert all('labels' not in doc for doc in encoded)
    truth = {str(doc['pmc_id']): relations(doc) for doc in read_jsonl(args.gold)}
    old = {str(doc['pmc_id']): relations(doc) for doc in read_jsonl(args.baseline)}
    by_doc = {doc['pmc_id']: doc for doc in encoded}
    grouped = defaultdict(list)
    counts = defaultdict(int)
    changes = {'union_add': defaultdict(int), 'replacement_add': defaultdict(int),
               'replacement_remove': defaultdict(int)}
    for row in scores:
        pmc, patient, concept = row['key']
        yes = concept in truth[pmc][patient]
        was = concept in old[pmc][patient]
        positive = row['logit'] > 0
        patient_count = sum(e['kind'] == 'patient' for e in by_doc[pmc]['entities'])
        grouped['all'].append((row['logit'], yes))
        grouped['one_patient' if patient_count == 1 else 'multiple_patients'].append((row['logit'], yes))
        counts['candidate_positive_truth'] += int(yes)
        counts['predicted_positive'] += int(positive)
        counts['tp' if yes else 'fp'] += int(positive)
        counts['fn'] += int(yes and not positive)
        counts['baseline_candidate_positive'] += int(was)
        if positive and not was:
            changes['union_add']['true' if yes else 'false'] += 1
            changes['replacement_add']['true' if yes else 'false'] += 1
        if was and not positive:
            changes['replacement_remove']['true' if yes else 'false'] += 1
    all_gold = sum(len(concepts) for doc in truth.values() for concepts in doc.values())
    result = {'scope': 'Post-two-frozen-F1-endpoints diagnostic; no fit, cutoff or epoch search; no new v2 calls',
              'quality_result_sha256': hashlib.sha256((Path(args.quality) / 'result.json').read_bytes()).hexdigest(),
              'all_gold_relations_including_candidate_outside': all_gold,
              'candidate_counts': dict(counts), 'fixed_endpoint_changes': changes,
              'ranking_diagnostic': {name: ranking(rows) for name, rows in grouped.items()}}
    write(args.output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
