"""Small synthetic boundary checks; no patient data, model or heldout gold."""
import copy
import json
from pathlib import Path
import sys
from assemble_public_phenotagger import assemble_document, sha


def main():
    text = 'Seizures. No ataxia. Long fingers and short stature.'
    doc = {'pmc_id': 'synthetic', 'full_text': [{'text': text, 'offset': 100, 'type': 'paragraph'}]}
    original = {'identifier': 'HP:0001250', 'offset': 100, 'length': 8, 'text': 'Seizures',
                'type': 'Phenotype', 'note': None}
    base = {'pmc_id': 'synthetic', 'entities': [original],
            'association': [{'patient_id': 'P1', 'phenotype': ['HP:0001250']}], 'extra': ['untouched']}
    def row(surface, ident, score=1):
        start = text.index(surface)
        return {'pmc_id': 'synthetic', 'passage_index': 0, 'passage_type': 'paragraph', 'passage_offset': 100,
                'local_start': start, 'local_end': start + len(surface), 'offset': 100 + start,
                'length': len(surface), 'text': surface, 'raw_hpo_id': ident, 'hpo_id': ident, 'score': score}
    candidates = [row('Seizures', 'HP:0001250'), row('ataxia', 'HP:0001251'),
                  row('Long fingers', 'HP:0001166'), row('fingers', 'HP:0001166'),
                  row('short stature', 'HP:0004322', .7)]
    original_base = copy.deepcopy(base)
    allowed = {'HP:0001250', 'HP:0001251', 'HP:0001166', 'HP:0004322'}
    pred, counts = assemble_document(doc, base, candidates, allowed)
    assert counts == {'native_candidates': 5, 'reject_overlap_original': 1, 'reject_negated': 1,
                      'reject_overlap_added': 1, 'added_entities': 2, 'added_relations': 0}
    assert pred['association'] == base['association'] and base == original_base
    assert {e['text'] for e in pred['entities']} == {'Seizures', 'Long fingers', 'short stature'}
    empty, empty_counts = assemble_document(doc, base, [], allowed)
    assert empty == base and empty_counts['added_entities'] == 0
    invalid = row('short stature', 'HP:0004322')
    invalid['offset'] += 1
    try:
        assemble_document(doc, base, [invalid], allowed)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched local/global offset was accepted')
    result = {'passed': True, 'gold_read': False, 'GPU_used': False,
              'source_sha256': sha(Path(__file__).with_name('assemble_public_phenotagger.py')),
              'check_sha256': sha(__file__), 'cases': ['base-overlap', 'fixed-negation', 'added-overlap',
              'native-decoded-no-new-threshold', 'relation-preserved', 'empty-document', 'offset-reject'],
              'counts': dict(counts)}
    with Path(sys.argv[1]).open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
