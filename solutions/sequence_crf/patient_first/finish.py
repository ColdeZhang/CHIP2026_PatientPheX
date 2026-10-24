"""Assemble strict final80 before gold; score exactly one new patient-first arm."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import signal
import sys
import time

from final_parser import parse_final
from patient_first import compact

COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')


def ref(path):
    path = Path(path); raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2); handle.write('\n')


def check_ref(item):
    actual = ref(item['path'])
    assert all(actual[key] == item[key] for key in ('bytes', 'sha256'))


def relations(doc):
    result = {}
    for association in doc['association']:
        pid = association['patient_id']; assert pid not in result
        result[pid] = {part for value in association['phenotype']
                       for part in (value.split(';') if COMPOUND.fullmatch(value) else [value])}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['assemble', 'evaluate'])
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(); b = args.directory; started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU endpoint >30s')))
    signal.alarm(30)
    plan = read(b/'endpoint-plan.json')
    assert plan['source_sha256'] == ref(__file__)['sha256']
    refs = plan['references']
    # The gold locator is separate and is never opened during assembly.
    for item in refs.values(): check_ref(item)
    assert plan['keep_rule'] == 'delta(micF1+macF1)>0 and delta(macF1)>=0'
    if args.mode == 'assemble':
        out = b/'assembled'; assert not out.exists()
        terminal = b/'inference-terminal'
        complete = read(terminal/'complete.json')
        assert complete['rows'] == 80 and complete['no_quality']
        response_path = terminal/'responses.jsonl'
        assert complete['responses_sha256'] == ref(response_path)['sha256']
        assert complete['manifest_sha256'] == ref(terminal/'responses.manifest.json')['sha256']
        manifest = read(terminal/'responses.manifest.json')
        assert manifest['strict_all80_EOS_and_PID_CID_schema'] and manifest['fallback_rows'] == 0
        assert manifest['fingerprint']['prompts_sha256'] == refs['prompts']['sha256']
        assert manifest['fingerprint']['final_parser_sha256'] == refs['parser']['sha256']
        assert manifest['fingerprint']['schema_sha256'] == refs['schema']['sha256']
        prompts, responses = rows(refs['prompts']['path']), rows(response_path)
        assert len(prompts) == len(responses) == 80
        by_id = {r['row_id']:r for r in responses}
        assert len(by_id) == 80 and set(by_id) == {r['row_id'] for r in prompts}
        audit = {r['row_id']:r['patient_first'] for r in read(refs['input_token_audit']['path'])}
        sys.path.insert(0, plan['lightweight_libraries'])
        import tokenizers
        assert tokenizers.__version__ == '0.23.1'
        tokenizer = tokenizers.Tokenizer.from_file(refs['tokenizer']['path'])
        old = rows(refs['baseline_pred']['path']); inputs = rows(refs['input']['path'])
        assert len(old) == len(inputs) == 16
        assert [d['pmc_id'] for d in old] == [d['pmc_id'] for d in inputs]
        new = copy.deepcopy(old); docs = {str(d['pmc_id']):d for d in new}
        decisions = {pmc:{str(a['patient_id']):set() for a in doc['association']} for pmc,doc in docs.items()}
        parsed = []; candidate_count = pair_count = 0
        for prompt in prompts:
            raw = by_id[prompt['row_id']]
            assert raw['input_tokens'] == audit[prompt['row_id']]['input_tokens']
            mapping = parse_final(raw, prompt)
            decoded = tokenizer.decode(raw['generated_token_ids'][:-1], skip_special_tokens=False)
            assert decoded == raw['response'], 'Actual generated IDs do not decode to the entire response'
            pmc = str(prompt['pmc_id'])
            assert set(prompt['patient_ids']) <= set(decisions[pmc])
            for code, patients in mapping.items():
                for patient in patients: decisions[pmc][patient].add(prompt['code_map'][code])
            candidate_count += len(prompt['code_map'])
            pair_count += len(prompt['code_map'])*len(prompt['patient_ids'])
            parsed.append(dict(row_id=prompt['row_id'], pmc_id=prompt['pmc_id'], candidate_to_patients=mapping))
        assert candidate_count == 587 and pair_count == 1557
        for before, after in zip(old, new):
            for association in after['association']:
                association['phenotype'] = sorted(decisions[str(after['pmc_id'])][str(association['patient_id'])])
            assert {k:v for k,v in before.items() if k != 'association'} == {k:v for k,v in after.items() if k != 'association'}
            assert [{k:v for k,v in a.items() if k != 'phenotype'} for a in before['association']] == [{k:v for k,v in a.items() if k != 'phenotype'} for a in after['association']]
        out.mkdir()
        with (out/'patient-first.jsonl').open('x') as handle:
            for doc in new: handle.write(json.dumps(doc, ensure_ascii=False)+'\n')
        save(out/'parsed-decisions.json', parsed)
        save(out/'freeze.json', dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
             plan=ref(b/'endpoint-plan.json'), prediction=ref(out/'patient-first.jsonl'),
             responses=ref(response_path), parsed=ref(out/'parsed-decisions.json'),
             inference_manifest=ref(terminal/'responses.manifest.json'), all80_strict=True,
             candidates=587, pairs=1557, all_nonassociation_exact=True, gold_read=False,
             no_baseline_fallback=True, seconds=time.monotonic()-started))
        print(json.dumps(read(out/'freeze.json'))); return
    freeze = read(b/'assembled/freeze.json')
    assert freeze['gold_read'] is False and freeze['all80_strict']
    for key in ('prediction', 'responses', 'parsed', 'inference_manifest'): check_ref(freeze[key])
    out = b/'evaluation'; out.mkdir(exist_ok=False)
    save(out/'before-gold.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
         freeze=ref(b/'assembled/freeze.json'), gold_not_yet_read=True))
    sys.path.insert(0, str(Path(refs['evaluator']['path']).parent))
    import evaluate as evaluator
    import validate_predictions as validator
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    check_ref(plan['gold'])
    gold = validator.load_jsonl(plan['gold']['path'])
    inputs = validator.load_jsonl(refs['input']['path'])
    prediction = validator.load_jsonl(freeze['prediction']['path'])
    assert {d['pmc_id'] for d in gold} == {d['pmc_id'] for d in inputs}
    hpo = validator.read_hpo(refs['hpo']['path'])
    assert hpo['version'] == 'hp/releases/2026-06-23'
    validation = validator.validate_predictions(inputs, prediction, hpo)
    save(out/'validation.json', validation); assert validation['valid']
    metrics = evaluator.evaluate_report(gold, prediction, 'all', 'text')
    save(out/'metrics.json', metrics)
    baseline = read(refs['baseline_metrics']['path'])
    delta = {key:metrics['metrics'][key]-value for key,value in baseline['metrics'].items()}
    assert delta['menF1'] == delta['docF1'] == 0
    goldmap = {d['pmc_id']:relations(d) for d in gold}
    oldmap = {d['pmc_id']:relations(d) for d in rows(refs['baseline_pred']['path'])}
    change = dict(added_tp=0, added_fp=0, removed_tp=0, removed_fp=0)
    for doc in prediction:
        current = relations(doc); old = oldmap[doc['pmc_id']]; truth = goldmap[doc['pmc_id']]
        assert set(current) == set(old)
        for pid in current:
            added = current[pid]-old[pid]; removed = old[pid]-current[pid]
            change['added_tp'] += len(added & truth[pid]); change['added_fp'] += len(added-truth[pid])
            change['removed_tp'] += len(removed & truth[pid]); change['removed_fp'] += len(removed-truth[pid])
    keep = delta['micF1']+delta['macF1'] > 0 and delta['macF1'] >= 0
    save(out/'results.json', dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
         metrics=metrics['metrics'], baseline=baseline['metrics'], baseline_reused_not_rescored=True,
         delta=delta, changes=change, keep=keep, keep_rule=plan['keep_rule'],
         scope='One final-epoch patient-first F1; same entities/pool. No union, scans, originaldev or official submission.',
         seconds=time.monotonic()-started))
    print(json.dumps(read(out/'results.json')))


if __name__ == '__main__':
    main()
