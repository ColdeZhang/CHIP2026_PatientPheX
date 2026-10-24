"""Replay frozen901 from four native response streams, without model or quality.

The Task1 template supplies only non-association bytes. All relations are rebuilt
from normal/reverse/public raw responses and the original patient limited codec.
Native historical counts/SHA checks are engineering checks, never score gates.
"""
import argparse
import ast
from collections import Counter, defaultdict
import copy
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import time
from types import SimpleNamespace


def ref(path):
    p = Path(path).resolve(); data = p.read_bytes()
    return dict(path=str(p), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(item):
    p = Path(item['path']); actual = ref(p)
    assert (actual['bytes'], actual['sha256']) == (item['bytes'], item['sha256']), p
    return p


def rows(path):
    return [json.loads(s) for s in Path(path).read_bytes().splitlines() if s.strip()]


def save(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    publish(path, raw)


def publish(path, raw):
    with Path(path).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def definitions(path, names, scope, constants=()):
    """Load named pure definitions, never execute a historical CLI or evaluator."""
    selected = []
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in constants:
                selected.append(node)
    got = {n.name for n in selected if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert got == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), scope)


def helpers(sources):
    scope = dict(json=json, re=re, copy=copy, deepcopy=deepcopy, Counter=Counter,
                 defaultdict=defaultdict, math=math, Path=Path)
    definitions(sources['original_parser'], ['parse_json'], scope)
    definitions(sources['reviewed_B_assembler'], ['parse_arm'], scope)
    definitions(sources['old_public_append'], ['append_keys'], scope)
    definitions(sources['reverse_union'], ['values', 'relations', 'union'], scope,
                constants=['COMPOUND'])
    definitions(sources['literal_helper'], ['field_span', 'associations', 'entity_support',
                                         'append_literal'], scope)
    definitions(sources['support_replacement'], ['replace_subset'], scope)
    definitions(sources['patient_schema'], ['ids', 'validate_input'], scope,
                constants=['INPUT_KEYS'])
    definitions(sources['patient_codec'], ['_Rejected', '_reject', '_unique_object',
                                         '_finite_float', 'classify_final'], scope,
                constants=['EOS_IDS'])
    return SimpleNamespace(**scope)


def keyed(prompts, inputs):
    by = {p['row_id']: p for p in prompts}
    assert len(by) == len(prompts)
    patients = {str(d['pmc_id']): {str(p['patient_id']) for p in d['patient']} for d in inputs}
    spaces = {did: set() for did in patients}
    for p in prompts:
        did = str(p['pmc_id']); assert did in patients and 'answer' not in p
        assert set(p['patient_ids']) == patients[did]
        assert all(isinstance(x, str) and x for x in p['code_map'].values())
        assert not spaces[did] & set(p['code_map'].values())
        spaces[did].update(p['code_map'].values())
    assert set(spaces) == set(patients)
    return by, dict(spaces)


def literal_splice(helper, line, associations):
    begin, end = helper.field_span(line, 'association')
    out = line[:begin] + json.dumps(associations, ensure_ascii=False) + line[end:]
    a, b = helper.field_span(out, 'association')
    assert (line[:begin], line[end:]) == (out[:a], out[b:])
    return out


def fingerprint_native(arm, refs, expected_rows):
    manifest = json.loads(refs[arm + '_manifest'].read_text())
    fp = manifest['fingerprint']; cfg = fp['configuration']
    assert manifest['output_sha256'] == ref(refs[arm + '_responses'])['sha256']
    assert manifest['expected_rows'] == expected_rows
    assert fp['prompts_sha256'] == ref(refs[arm + '_prompts'])['sha256']
    assert cfg['max_input_tokens'] == 14000 and cfg['max_new_tokens'] == 1600
    assert fp['source_manifest_sha256'] == '74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
    if arm != 'patient':
        assert fp['source_sha256'] == '33a5a497ff8106a02cab5dda5d59193b4b1f14aed4d5791464d266b7d05d5669'
        assert fp['adapter_files']['adapter_model.safetensors'] == '1a703b244b17c1378b383e4d6feefe2486b1a888e78cbd6f9f25c3fa6e46f45b'
        assert manifest['base_parameters'] == 8953803264
        assert manifest['total_parameters'] == 8997081600
    else:
        assert fp['limited_codec_sha256'] == '60731e1297930415afd8bb8c2a979814d7c8cb9a8c132c6c6debbd85d50cb289'
        assert fp['schema_sha256'] == '25802e682fcd801351377ce3a9f1815a2433c44e8d73c50ab35fba5016358f08'
        assert fp['adapter_files']['adapter_model.safetensors'] == 'f787f18de7843f4af092e0a36ab5df05e9109081c1d6df81b7c3d9cb864065b7'
    return manifest


def assemble(refs, helper):
    """Fixed recipe: replace old public gate, then add P minus old-gate a179."""
    inputs = rows(refs['input']); old = rows(refs['old_candidates'])
    task_lines = refs['task1_template'].read_bytes().decode().splitlines(keepends=True)
    task = [json.loads(s) for s in task_lines]
    assert len(inputs) == len(old) == len(task) == 16
    assert sum(len(d['patient']) for d in inputs) == 36
    assert sum(len(d['entities']) for d in task) == 1061
    assert [str(d['pmc_id']) for d in inputs] == [str(d['pmc_id']) for d in task]
    assert all(set(d) == {'pmc_id', 'pmid', 'patient', 'full_text'} for d in inputs)
    old_by = {str(d['pmc_id']): d for d in old}
    assert set(old_by) == {str(d['pmc_id']) for d in inputs}
    streams = {}; spaces = {}; stats = {}; manifests = {}
    for arm, count in [('normal', 64), ('reverse', 64), ('public', 42), ('patient', 64)]:
        prompts = rows(refs[arm + '_prompts']); raw = rows(refs[arm + '_responses'])
        assert len(prompts) == len(raw) == count
        by_id, spaces[arm] = keyed(prompts, inputs)
        assert len({r['row_id'] for r in raw}) == count
        assert {r['row_id'] for r in raw} == set(by_id)
        streams[arm] = (prompts, raw, by_id)
        manifests[arm] = fingerprint_native(arm, refs, count)
    assert spaces['normal'] == spaces['reverse'] == spaces['patient']
    assert sum(map(len, spaces['normal'].values())) == 466
    assert sum(map(len, spaces['public'].values())) == 276
    for did in spaces['normal']:
        assert not spaces['normal'][did] & spaces['public'][did]
    normal, stats['normal'] = helper.parse_arm(*streams['normal'][:2], old_by, helper.parse_json, True)
    reverse, stats['reverse'] = helper.parse_arm(*streams['reverse'][:2], old_by, helper.parse_json, True)
    public, stats['public'] = helper.parse_arm(*streams['public'][:2], old_by, helper.parse_json, False)
    # Old support uses the exact historical HGB6401 entity set, not new LGB1061.
    old_support = helper.entity_support(rows(refs['old_HGB_support']))
    old_public = {k for k in public if (k[0], k[2]) in old_support}
    new_support = helper.entity_support(task)
    new_public = {k for k in public if (k[0], k[2]) in new_support}
    fresh_normal = []
    for source, template in zip(inputs, task):
        did = str(source['pmc_id']); doc = deepcopy(template)
        # Deliberately discard template associations; only raw decisions fill these.
        doc['association'] = [dict(patient_id=p['patient_id'], phenotype=sorted(
            c for d, pid, c in normal if d == did and pid == str(p['patient_id'])))
            for p in source['patient']]
        fresh_normal.append(doc)
    supported_normal = helper.append_keys(fresh_normal, old_public)
    a179 = []
    for doc in supported_normal:
        reverse_doc = deepcopy(doc); did = str(doc['pmc_id'])
        for association in reverse_doc['association']:
            association['phenotype'] = sorted(c for d, p, c in reverse
                                               if d == did and p == str(association['patient_id']))
        a179.append(helper.union(doc, reverse_doc)[0])
    old_keys = helper.associations(a179)
    native0e2c = ''.join(literal_splice(helper, line, doc['association'])
                       for line, doc in zip(task_lines, a179)).encode()
    removed, added = old_public - new_public, new_public - old_public
    base_lines = native0e2c.decode().splitlines(keepends=True)
    supported = ''.join(helper.replace_subset(helper, line, removed, added)
                        for line in base_lines).encode()
    # Re-run exact limited codec, including partial-PID abstention and invalid-row
    # whole abstention. Do not replace it with the stricter fulltext CID codec.
    patient = set(); decisions = []; patient_stats = Counter(complete=0, partial=0, abstain=0)
    for record in streams['patient'][1]:
        prompt = streams['patient'][2][record['row_id']]
        decision = helper.classify_final(record, prompt)
        decisions.append(dict(row_id=record['row_id'], **decision))
        patient_stats[decision['status']] += 1
        patient.update((str(prompt['pmc_id']), pid, prompt['code_map'][cid])
                       for pid, cid in decision['selected_pairs'])
    assert decisions == json.loads(refs['patient_saved_decisions'].read_text())
    assert dict(patient_stats) == manifests['patient']['codec_status_counts']
    # Preserve historical delta policy explicitly; never substitute unconditional P union.
    patient_delta = patient - old_keys
    after_support = helper.associations([json.loads(s) for s in supported.splitlines()])
    actual_added = patient_delta - after_support
    final = ''.join(helper.append_literal(line, actual_added)
                    for line in supported.decode().splitlines(keepends=True)).encode()
    final_docs = [json.loads(s) for s in final.splitlines()]
    assert helper.associations(final_docs) == (old_keys - old_public) | new_public | patient_delta
    for before, after in zip(task_lines, final.decode().splitlines(keepends=True)):
        x, y = helper.field_span(before, 'association'); u, v = helper.field_span(after, 'association')
        assert (before[:x], before[y:]) == (after[:u], after[v:])
    keys = dict(normal=normal, reverse=reverse, public_raw=public, old_public=old_public,
                new_public=new_public, a179=old_keys, patient=patient,
                patient_delta=patient_delta, actual_patient_added=actual_added,
                final=helper.associations(final_docs))
    return dict(prediction=final, reconstructed0e2c=native0e2c, reconstructed3049=supported,
                keys=keys, parser_statistics=stats, patient_codec_statistics=dict(patient_stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output-directory', type=Path, required=True)
    args = ap.parse_args(); started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU replay exceeded60s')))
    signal.alarm(60)
    plan = json.loads(args.plan.read_text())
    assert plan['policy'] == 'native-four-stream-901-byteexact-replay-v1'
    assert plan['budget_seconds'] == 60 and plan['attempts'] == 1
    checked(plan['source']); assert plan['source']['sha256'] == ref(__file__)['sha256']
    checked(plan['contract']); contract = json.loads(Path(plan['contract']['path']).read_text())
    assert contract['policy'] == plan['policy'] and contract['new_quality_calls'] == 0
    refs = {key: checked(item) for key, item in plan['references'].items()}
    sources = {key: checked(item) for key, item in plan['pure_sources'].items()}
    assert plan['references'] == contract['references'] and plan['pure_sources'] == contract['pure_sources']
    output = args.output_directory.resolve(); output.mkdir(parents=True, exist_ok=False)
    save(output / 'intent.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
         plan=ref(args.plan), source=ref(__file__), attempts=1, gold_reads=0, quality_calls=0))
    try:
        result = assemble(refs, helpers(sources))
        checks = {}
        for key, expected in contract['expected_bytes'].items():
            raw = result[key]; observed = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            checks[key] = dict(expected=expected, actual=observed, exact=observed == expected)
        publish(output / 'prediction.jsonl', result['prediction'])
        counts = {key: len(value) for key, value in result['keys'].items()}
        # Preserve observations before assertions, including a first failure.
        save(output / 'observations.json', dict(byte_checks=checks, counts=counts,
             parser_statistics=result['parser_statistics'], patient_codec_statistics=result['patient_codec_statistics']))
        assert all(v['exact'] for v in checks.values()), checks
        assert counts == contract['expected_counts'], counts
        for key, value in result['keys'].items():
            save(output / (key + '.keys.json'), sorted(value))
        report = dict(status='PASS_NATIVE_FOUR_STREAM_901_BYTEEXACT_NO_QUALITY',
            plan=ref(args.plan), source=ref(__file__), contract=plan['contract'],
            prediction=ref(output / 'prediction.jsonl'), counts=counts,
            byte_checks=checks, documents=16, patients=36, entities=1061,
            all_nonassociation_literal_bytes_preserved=True,
            original_limited_patient_codec_replayed=True,
            patient_policy='patient positives minus reconstructed old-gate a179, then append to new-support baseline',
            no_901_or3049_prediction_used_as_generation_input=True,
            template_associations_discarded_before_generation=True,
            scope='Frozen originaldev inputs/candidates/Task1 entities and native raw responses; not raw-text/model reproduction.',
            fit_calls=0, NN_calls=0, gold_reads=0, quality_calls=0,
            seconds=time.monotonic()-started)
        save(output / 'reproduction.json', report)
        print(json.dumps(dict(status=report['status'], prediction=report['prediction'], seconds=report['seconds'])))
    except BaseException as error:
        save(output / 'failure.json', dict(error_type=type(error).__name__, message=str(error)[:4000],
             elapsed_seconds=time.monotonic()-started, rerun_authorized=False,
             gold_reads=0, quality_calls=0, fit_calls=0, NN_calls=0))
        raise
    finally:
        fd = os.open(output, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        signal.alarm(0)


if __name__ == '__main__':
    main()
