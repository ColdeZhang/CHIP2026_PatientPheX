#!/usr/bin/env python3
"""Bounded inference contracts using cached input; no fit or quality evaluation."""
import argparse
import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import predict_existing_selector as target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, help='Successful load-only replay job JSON.')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    command = json.loads(Path(args.job).read_text())['command']
    options = dict(zip(command[2::2], command[3::2]))
    inputs = target.read_rows(options['--input'])
    bases = target.read_rows(options['--base'])
    pools = target.read_rows(options['--candidates'])
    associations = target.read_rows(options['--associations'])
    did = next(iter(inputs))
    documents = {did: inputs[did]}
    baseline = {did: bases[did]}
    candidates = {did: pools[did]}
    related = {did: associations[did]}
    target.validate_sources(documents, baseline, candidates, related)
    checks = []

    def rejects(name, mutate):
        values = deepcopy([documents, baseline, candidates, related])
        mutate(*values)
        try:
            target.validate_sources(*values)
        except ValueError:
            checks.append(name)
        else:
            raise AssertionError('unexpected acceptance: ' + name)

    rejects('input_annotations_rejected', lambda d, b, c, a: d[did].update(entities=[]))
    rejects('missing_candidate_document_rejected', lambda d, b, c, a: c.clear())
    rejects('duplicate_candidate_key_rejected', lambda d, b, c, a: c[did]['candidates'].append(c[did]['candidates'][0]))
    rejects('cross_passage_coordinate_rejected', lambda d, b, c, a: c[did]['candidates'][0].update(offset=10**9))
    rejects('wrong_text_rejected', lambda d, b, c, a: c[did]['candidates'][0].update(text='not the source text'))
    rejects('nonfinite_score_rejected', lambda d, b, c, a: c[did]['candidates'][0].update(probability=float('nan')))
    rejects('missing_source_flag_rejected', lambda d, b, c, a: c[did]['candidates'][0].pop('bioformer_present'))
    rejects('wrong_patient_keys_rejected', lambda d, b, c, a: a[did]['association'].append({'patient_id': 'not-an-input-patient', 'phenotype': []}))
    rejects('wrong_identifier_rejected', lambda d, b, c, a: c[did]['candidates'][0].update(identifier='HP:wrong'))
    duplicate = out/'duplicate-documents.jsonl'
    duplicate.write_text(json.dumps(inputs[did])+'\n'+json.dumps(inputs[did])+'\n')
    try:
        target.read_rows(duplicate)
    except ValueError:
        checks.append('duplicate_document_rejected')
    else:
        raise AssertionError('duplicate document accepted')

    def decode_ast(path):
        return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body
                             if isinstance(n, ast.FunctionDef) and n.name == 'decode'), include_attributes=False)
    assert decode_ast(target.__file__) == decode_ast(Path(target.__file__).with_name('calibrate_public_selector_cut.py'))
    checks.append('decoder_AST_identical_to_frozen_function')
    # Every cached selector probability becomes deliberately nonnumeric. New
    # inference must use the loaded model, not trust or validate those values.
    altered = deepcopy(candidates[did])
    for c in altered['candidates']:
        c['selector_probability'] = 'ignored cached value'
    new_id, empty_id = 'unseen-document-id', 'empty-candidate-document'
    fixture = {
        '--input': [dict(inputs[did], pmc_id=new_id), dict(inputs[did], pmc_id=empty_id)],
        '--base': [dict(bases[did], pmc_id=new_id), dict(bases[did], pmc_id=empty_id)],
        '--candidates': [dict(altered, pmc_id=new_id), {'pmc_id': empty_id, 'candidates': []}],
    }
    for flag, rows in fixture.items():
        path = out/(flag[2:]+'.jsonl')
        path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows))
        options[flag] = str(path)
    options.pop('--associations')
    options['--output-dir'] = str(out/'task1-only')
    calls = []
    original_predict = target.HistGradientBoostingClassifier.predict_proba

    def no_fit(*a, **kw):
        raise AssertionError('inference tried to fit')

    def predict(model, matrix, *a, **kw):
        calls.append(len(matrix))
        return original_predict(model, matrix, *a, **kw)

    target.HistGradientBoostingClassifier.fit = no_fit
    target.DictVectorizer.fit = no_fit
    target.DictVectorizer.fit_transform = no_fit
    target.HistGradientBoostingClassifier.predict_proba = predict
    # Explicitly forbid opening known annotation training/evaluation files.
    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            name = str(arguments[0])
            if name.endswith('.gold.jsonl') or name.endswith('/train64.jsonl') or name.endswith('/train48.jsonl'):
                raise AssertionError('annotation file opened during inference')
    sys.addaudithook(audit)
    sys.argv = [target.__file__] + [item for pair in options.items() for item in pair]
    target.main()
    actual = target.read_rows(out/'task1-only/selector-scores.jsonl')
    assert actual[new_id]['candidates'] == pools[did]['candidates']
    predictions = target.read_rows(out/'task1-only/predictions.jsonl')
    assert all('association' not in row for row in predictions.values())
    assert predictions[empty_id]['entities'] == [] and actual[empty_id]['candidates'] == []
    assert calls == [len(pools[did]['candidates'])]
    checks.extend(['arbitrary_document_ids_accepted', 'empty_candidate_document_supported',
                   'stored_probabilities_ignored_actual_predict_called', 'Task1_only_omits_associations',
                   'fit_and_annotation_IO_traps_not_triggered'])
    report = {'passed': True, 'checks': checks, 'source_sha256': target.sha(target.__file__),
              'checker_sha256': target.sha(__file__), 'job_sha256': target.sha(args.job),
              'prediction_calls': calls, 'new_model_fit': False, 'quality_evaluation': False,
              'synthetic_input_from_cached_text': True, 'actual_original_probabilities_recovered': True}
    (out/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
