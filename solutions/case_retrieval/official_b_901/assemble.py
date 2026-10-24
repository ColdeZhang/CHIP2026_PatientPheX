"""Assemble the frozen901 B policy only after actual canonical-space proof.

No model, fitting, labels or evaluator. Complete normal/reverse rows retain the
original CID fallback; public-new invalid CIDs abstain. Patient positives arrive
from the separately frozen original limited codec, with no baseline fallback.
"""
import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time

PINS = {
    'input_contract': '2c1690660c58337235e9e11d96633109ac3ac3ffd62fa5e0600980c4b8a2ba21',
    'input': '878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4',
    'hpo': 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',
    'original_base': '41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0',
    'original_merger': '817fdc4494e7ed349085fe97c5744409d66d0f1de476089ac8c5696a84a6b4a1',
    'literal_helper': '651f9dcc592ebc7fff40d0c031db708d69c13546ca602e8b535d9f39916c8b38',
}


def ref(path):
    p = Path(path)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def functions(path, names, scope):
    nodes = [n for n in ast.parse(Path(path).read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)


def parse_arm(prompts, responses, old, parse_json, fallback):
    """Original merger CID resolution semantics, complete response coverage first."""
    by_id = {r['row_id']: r for r in responses}
    assert len(by_id) == len(responses) == len(prompts)
    assert set(by_id) == {p['row_id'] for p in prompts}
    result, statistics = set(), Counter()
    seen = set()
    for prompt in prompts:
        response = by_id[prompt['row_id']]
        pmc = str(prompt['pmc_id'])
        assert str(response['pmc_id']) == pmc and isinstance(response['response'], str)
        parsed = parse_json(response['response']) if not response.get('hit_token_limit', False) else None
        patients = set(prompt['patient_ids'])
        for code, concept in prompt['code_map'].items():
            assert (pmc, concept) not in seen
            seen.add((pmc, concept))
            value = parsed.get(code) if isinstance(parsed, dict) else None
            if not isinstance(value, list) or any(str(p) not in patients for p in value):
                statistics['unresolved_candidate_decisions'] += 1
                if fallback:
                    for previous in old[pmc]['association']:
                        if concept in previous.get('phenotype', []):
                            result.add((pmc, str(previous['patient_id']), concept))
                continue
            statistics['valid_candidate_decisions'] += 1
            for pid in value:
                result.add((pmc, str(pid), concept))
    return result, dict(statistics)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan', type=Path, required=True)
    ap.add_argument('--output-directory', type=Path, required=True)
    a = ap.parse_args(); began = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Assembly exceeded60s')))
    signal.alarm(60)
    plan = json.loads(a.plan.read_text())
    assert plan['policy'] == 'frozen901_officialB_only_no_quality_v1'
    assert plan['source'] == ref(__file__)
    refs = plan['references']
    for name, expected in refs.items():
        assert ref(expected['path']) == expected, name
        if name in PINS: assert expected['sha256'] == PINS[name], name
    assert set(PINS) <= set(refs)
    get = lambda name: refs[name]['path']
    source = dict(Path=Path, re=re, json=json, defaultdict=defaultdict, deepcopy=deepcopy)
    functions(get('original_base'), ['parse_hpo', 'association_keys'], source)
    functions(get('original_merger'), ['parse_json'], source)
    functions(get('literal_helper'), ['entity_support', 'field_span', 'append_literal', 'associations'], source)
    _, allowed, alt, hpo_version = source['parse_hpo'](get('hpo'))
    inputs = rows(get('input')); by = {str(d['pmc_id']): d for d in inputs}
    old_rows = rows(get('old_candidates')); old = {str(d['pmc_id']): d for d in old_rows}
    task_lines = Path(get('task1_prediction')).read_text().splitlines(keepends=True)
    task = [json.loads(line) for line in task_lines]
    assert len(by) == len(inputs) == len(old_rows) == len(task) == 100
    assert set(by) == set(old) == {str(d['pmc_id']) for d in task}
    assert sum(len(d['patient']) for d in inputs) == 244
    assert sum(len(x['phenotype']) for d in task for x in d['association']) == 0
    assert [str(d['pmc_id']) for d in task] == [str(d['pmc_id']) for d in inputs]
    task_report=json.loads(Path(get('task1_report')).read_text())
    assert task_report['status']=='frozen_LightGBM103_application_complete'
    assert task_report['prediction']==refs['task1_prediction']
    assert task_report['source']['sha256']=='a9b7c19901f9fd1a6714a17bbdf99cf0262703ed89d9d563b961ca41c0d097ff'
    assert task_report['inputs']['model']['sha256']=='803b646771d5d7701730ce2a7161275158e63b63ddec36fed9ce793b64e6f98d'
    assert task_report['inputs']['input']==refs['input'] and task_report['documents']==100
    for current, original in zip(task, inputs):
        assert current['pmid']==original.get('pmid')
        assert [p['patient_id'] for p in current['association']]==[p['patient_id'] for p in original['patient']]
    patients = {did: {str(p['patient_id']) for p in d['patient']} for did, d in by.items()}
    old_raw, old_canonical = {}, {}
    for did, d in old.items():
        old_raw[did], old_canonical[did] = set(), set()
        for entity in d['entities']:
            keys = source['association_keys'](entity); old_raw[did].update(keys)
            if str(entity['identifier']) != '-1':
                assert all(re.fullmatch(r'HP:[0-9]{7}', key) for key in keys)
                old_canonical[did].update(alt.get(key, key) for key in keys)
        assert {str(x['patient_id']) for x in d['association']} == patients[did]
    arm_prompts = {name: rows(get(name + '_prompts')) for name in ('normal', 'reverse', 'public', 'patient')}
    for arm in ('normal','reverse','public'):
        manifest=json.loads(Path(get(arm+'_response_manifest')).read_text())
        assert manifest['output_sha256']==refs[arm+'_responses']['sha256']
        assert manifest['expected_rows']==manifest['native_input_rows_verified']==len(arm_prompts[arm])
        finger=manifest['fingerprint']
        assert finger['prompts_sha256']==refs[arm+'_prompts']['sha256']
        assert finger['source_sha256']=='fe98e76d3692401df9207f67ace24cef57249944ef846b08bc45f8675e0e1333'
        assert finger['source_manifest_sha256']=='74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7'
        assert finger['adapter_files']['adapter_model.safetensors']=='1a703b244b17c1378b383e4d6feefe2486b1a888e78cbd6f9f25c3fa6e46f45b'
        assert manifest['total_parameters']==8997081600
    maps = {}
    for arm, prompts in arm_prompts.items():
        assert len({p['row_id'] for p in prompts}) == len(prompts)
        values = defaultdict(set)
        for p in prompts:
            did = str(p['pmc_id']); assert did in by
            assert set(p['patient_ids']) == patients[did]
            assert all(isinstance(v, str) for v in p['code_map'].values())
            assert not values[did] & set(p['code_map'].values())
            values[did].update(p['code_map'].values())
        maps[arm] = {did: values[did] for did in by}
    assert maps['normal'] == maps['reverse'] == maps['patient'] == old_raw
    spaces = json.loads(Path(get('candidate_space')).read_text())
    assert spaces['status'] == 'OLD_VS_PUBLIC_NEW_CANONICAL_DISJOINT_ALL_B_DOCUMENTS'
    assert set(spaces['documents']) == set(by)
    actual_space = {}
    for did in by:
        public = maps['public'][did]
        assert public == set(spaces['documents'][did]['new_public_canonical_hpo'])
        assert all(c in allowed and alt.get(c, c) == c for c in public)
        # Also protect a literal NIL string coinciding with an HPO string, since
        # final association values themselves have no separate type field.
        patient_as_hpo = {alt.get(v, v) for v in old_raw[did] if re.fullmatch(r'HP:[0-9]{7}', v)}
        assert not public & (old_canonical[did] | patient_as_hpo | old_raw[did]), did
        actual_space[did] = dict(old_concepts=len(old_raw[did]), public_concepts=len(public),
                                canonical_intersection=0, raw_string_intersection=0)
    patient_list = json.loads(Path(get('patient_positive_keys')).read_text())
    patient = {tuple(key) for key in patient_list}
    assert len(patient) == len(patient_list)
    for did, pid, concept in patient:
        assert did in by and pid in patients[did] and concept in maps['patient'][did]
        assert concept not in maps['public'][did]
    normal, normal_stats = parse_arm(arm_prompts['normal'], rows(get('normal_responses')), old, source['parse_json'], True)
    reverse, reverse_stats = parse_arm(arm_prompts['reverse'], rows(get('reverse_responses')), old, source['parse_json'], True)
    public_raw, public_stats = parse_arm(arm_prompts['public'], rows(get('public_responses')), old, source['parse_json'], False)
    support = source['entity_support'](task)
    public_supported = {key for key in public_raw if (key[0], key[2]) in support}
    # Let O=N union R and G_old subset public_new. Since P intersect public_new
    # is empty, (O union G_new) union (P minus (O union G_old)) == O union G_new union P.
    assert not {(d, c) for d, _, c in patient} & {(d, c) for d, _, c in public_raw}
    union = normal | reverse | public_supported | patient
    for did, pid, concept in union:
        assert did in by and pid in patients[did] and isinstance(concept, str)
    out = a.output_directory.resolve(); out.mkdir(parents=True, exist_ok=False)
    # Starting from an empty association template means this deterministic final
    # sort may differ from historical intermediate append order. Relation SETS,
    # accepted entities and all other literal fields are the frozen method.
    payload = ''.join(source['append_literal'](line, union) for line in task_lines).encode()
    prediction = out / 'prediction.jsonl'
    with prediction.open('xb') as f: f.write(payload); f.flush(); os.fsync(f.fileno())
    predicted = rows(prediction)
    assert source['associations'](predicted) == union
    assert [d['entities'] for d in predicted] == [d['entities'] for d in task]
    for line, final_line in zip(task_lines, payload.decode().splitlines(keepends=True)):
        x, y = source['field_span'](line, 'association'); u, v = source['field_span'](final_line, 'association')
        assert (line[:x], line[y:]) == (final_line[:u], final_line[v:])
    for name, keys in [('normal', normal), ('reverse', reverse), ('public-raw', public_raw),
                       ('public-supported', public_supported), ('patient', patient)]:
        save(out / (name + '.keys.json'), sorted(keys))
    save(out / 'canonical-space-proof.json', dict(status='ALL100_ACTUAL_PATIENT_OLD_PUBLIC_SPACES_DISJOINT',
         documents=actual_space, patient_positive_pairs=len(patient), positive_public_intersection=0,
         hpo_version=hpo_version, old_counterfactual_gate_omitted_by_set_identity=True,
         set_identity='(O union Gnew) union (P minus (O union Gold)) = O union Gnew union P because P intersect public_new=empty',
         no_dev_specific_keys=True, literal_HPO_text_collisions_also_checked=True))
    report = dict(status='COMPLETE_FROZEN901_B_ASSEMBLY_NO_QUALITY', at_utc=datetime.now(timezone.utc).isoformat(),
        source=ref(__file__), plan=ref(a.plan), references=refs, prediction=ref(prediction),
        documents=100, patients=244, entities=sum(len(d['entities']) for d in predicted),
        relations=len(union), counts=dict(normal=len(normal), reverse=len(reverse), public_raw=len(public_raw),
        public_supported=len(public_supported), patient=len(patient), patient_new=len(patient-(normal|reverse|public_supported))),
        parser_statistics=dict(normal=normal_stats, reverse=reverse_stats, public_new=public_stats),
        entities_and_nonassociation_literal_bytes_preserved=True, original_normal_reverse_fallback=True,
        public_and_patient_no_fallback=True, old_gate_omitted_only_after_actual_space_proof=True,
        metadata_provenance='Each bound source/response manifest is separately reviewed; this assembler validates actual schema, input and set identity, not upstream neural execution.',
        fit_calls=0, NN_calls=0, gold_reads=0, quality_calls=0, seconds=time.monotonic()-began,
        artifacts={p.name: ref(p) for p in out.iterdir() if p.is_file()})
    save(out / 'assembly.json', report)
    fd=os.open(out, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    signal.alarm(0)
    print(json.dumps({k: report[k] for k in ('status','seconds','prediction','counts')}))


if __name__ == '__main__': main()
