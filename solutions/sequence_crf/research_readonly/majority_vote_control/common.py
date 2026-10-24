"""Frozen cache-only N/R/P majority; no model or evaluator imports here."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

SCHEMA = 'normal-reverse-patient-majority-v1'
KEEP = 'both folds Score strictly greater than their cached triple-union Score; macro disclosed without veto'
METHOD = '9016863cb3bff53da5dfde25fce634bd60882f057d0c14501be7cb4937e71c67'
B_INPUT = '878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4'
COMPOUND = re.compile(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+')
METRICS = ('menF1', 'docF1', 'micF1', 'macF1', 'Score')


def now():
    return datetime.now(timezone.utc).isoformat()


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(value):
    assert ref(value['path']) == value, value['path']
    return Path(value['path'])


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, data):
    path = Path(path); temporary = path.with_name(path.name + '.pending')
    with temporary.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()
    for directory in (path.parent, path.parent.parent):
        fd = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def save(path, value):
    publish(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def load_module(name, value):
    spec = importlib.util.spec_from_file_location(name, checked(value))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def gold_guard(plan):
    """The only transition enabling a gold open is inside the gated score worker."""
    state = {'enabled': False}
    paths = {str(Path(f['gold']['path']).resolve()) for f in plan['folds'].values()}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.realpath(os.fsdecode(args[0]))
            if path in paths and not state['enabled']:
                raise PermissionError('Gold access blocked before accepted receipt and common freeze')
    sys.addaudithook(audit)
    return state


def load_plan(path):
    assert __debug__, 'Assertions must remain enabled'
    plan = read(path); guard = gold_guard(plan)
    assert plan['schema'] == SCHEMA and plan['fold_order'] == ['F1', 'F0']
    assert plan['vote_threshold'] == 2 and plan['members'] == ['normal', 'reverse', 'patient']
    assert plan['keep_rule'] == KEEP and plan['new_evaluator_calls'] == 2
    assert plan['score_budget_seconds'] == 30 and plan['term_grace_seconds'] == 1
    for name, value in plan['sources'].items():
        assert Path(value['path']).name == name; checked(value)
    assert plan['sources']['common.py'] == ref(__file__)
    for value in plan['references'].values(): checked(value)
    original = read(plan['references']['triple_plan']['path'])
    original_freeze = read(plan['references']['triple_freeze']['path'])
    for fold in plan['fold_order']:
        spec = plan['folds'][fold]
        for name, previous in [('normal','old'),('reverse','reverse'),('patient','patient'),('input','input')]:
            assert spec['references'][name] == original['folds'][fold]['references'][previous]
        assert spec['gold'] == original['folds'][fold]['gold']  # metadata only
        assert spec['references']['baseline'] == original_freeze['predictions'][fold]
        for value in plan['folds'][fold]['references'].values(): checked(value)
        metric = read(plan['folds'][fold]['references']['baseline_metrics']['path'])
        assert metric['evaluator_version'] == 'patientphex-public-approx-v2'
        assert metric['configuration']['body_scope'] == 'all'
        assert metric['configuration']['unlinked_document_mode'] == 'text'
        assert metric['prediction_sha256'] == plan['folds'][fold]['references']['baseline']['sha256']
        assert metric['gold_sha256'] == plan['folds'][fold]['gold']['sha256']
    return plan, guard


def rows(path):
    lines = Path(path).read_bytes().decode('utf-8').splitlines(keepends=True)
    docs = [json.loads(line) for line in lines]
    ids = [str(d['pmc_id']) for d in docs]
    assert len(ids) == len(set(ids))
    return lines, docs, ids


def literal_keys(docs):
    result = set()
    for doc in docs:
        pids = set()
        for association in doc['association']:
            pid = association['patient_id']; assert pid not in pids
            pids.add(pid); values = association['phenotype']
            assert all(isinstance(v, str) for v in values)
            assert len(values) == len(set(values)), 'Duplicate votes must not count'
            result.update((str(doc['pmc_id']), pid, value) for value in values)
    return result


def evaluation_keys(docs):
    return {(did, pid, item) for did, pid, value in literal_keys(docs)
            for item in (value.split(';') if COMPOUND.fullmatch(value) else [value])}


def field_span(line, wanted):
    """Adapted from audited literal helper 651f9dcc, rejecting duplicate keys."""
    decoder = json.JSONDecoder()
    def space(pos):
        while pos < len(line) and line[pos].isspace(): pos += 1
        return pos
    pos = space(0); assert line[pos] == '{'; pos += 1
    found = None; names = set()
    while True:
        pos = space(pos)
        if line[pos] == '}': break
        name, pos = decoder.raw_decode(line, pos)
        assert isinstance(name, str) and name not in names; names.add(name)
        pos = space(pos); assert line[pos] == ':'; pos = space(pos + 1)
        begin = pos; _, pos = decoder.raw_decode(line, pos)
        if name == wanted: found = begin, pos
        pos = space(pos)
        if line[pos] == '}': break
        assert line[pos] == ','; pos += 1
    assert not line[pos + 1:].strip() and found is not None
    return found


def array_spans(text):
    decoder = json.JSONDecoder(); pos = 1; result = []
    assert text[0] == '[' and text[-1] == ']'
    while True:
        while text[pos].isspace(): pos += 1
        if text[pos] == ']': break
        begin = pos; _, pos = decoder.raw_decode(text, pos); result.append((begin, pos))
        while text[pos].isspace(): pos += 1
        if text[pos] == ']': break
        assert text[pos] == ','; pos += 1
    assert pos == len(text)-1
    return result


def phenotype_spans(line):
    start, end = field_span(line, 'association')
    spans = []
    for begin, finish in array_spans(line[start:end]):
        a, b = field_span(line[start+begin:start+finish], 'phenotype')
        spans.append((start+begin+a, start+begin+b))
    return spans


def mask_values(line):
    for begin, end in reversed(phenotype_spans(line)):
        line = line[:begin] + '[]' + line[end:]
    return line.encode('utf-8')


def filter_line(line, selected):
    old = json.loads(line); replacements = []
    spans = phenotype_spans(line)
    assert len(spans) == len(old['association'])
    for association, (begin, end) in zip(old['association'], spans):
        kept = [value for value in association['phenotype']
                if (str(old['pmc_id']), association['patient_id'], value) in selected]
        if kept != association['phenotype']:
            replacements.append((begin, end, json.dumps(kept, ensure_ascii=False)))
    result = line
    for begin, end, value in reversed(replacements): result = result[:begin]+value+result[end:]
    assert mask_values(line) == mask_values(result), 'Every byte outside phenotype values must stay exact'
    return result


def inspect_fold(spec):
    refs = spec['references']; expected = spec['expected']
    lines, baseline, ids = rows(refs['baseline']['path'])
    _, inputs, input_ids = rows(refs['input']['path'])
    assert ids == input_ids and len(ids) == expected['documents'] == 16
    assert all(set(d) == {'pmc_id', 'pmid', 'patient', 'full_text'} for d in inputs)
    assert sum(len(d['entities']) for d in baseline) == expected['entities']
    assert sum(len(d['association']) for d in baseline) == expected['patients']
    for before, inp in zip(baseline, inputs):
        assert [a['patient_id'] for a in before['association']] == [a['patient_id'] for a in inp['patient']]
    members = {}
    for name in ('normal', 'reverse', 'patient'):
        _, docs, member_ids = rows(refs[name]['path']); assert member_ids == ids
        for before, member in zip(baseline, docs):
            assert {k:v for k,v in before.items() if k != 'association'} == {k:v for k,v in member.items() if k != 'association'}
            assert [{k:v for k,v in a.items() if k != 'phenotype'} for a in before['association']] == [{k:v for k,v in a.items() if k != 'phenotype'} for a in member['association']]
        members[name] = literal_keys(docs)
    base_keys = literal_keys(baseline)
    assert base_keys == set.union(*members.values())
    assert len(base_keys) == expected['baseline_relations']
    return lines, baseline, members


def vote(members):
    assert set(members) == {'normal', 'reverse', 'patient'}
    counts = Counter(key for values in members.values() for key in values)
    return {key for key, count in counts.items() if count >= 2}, counts


def check_acceptance(path, expected_sha):
    actual = ref(path); assert actual['sha256'] == expected_sha
    gate = read(path)
    assert gate['schema'] == 'official-901-accepted-release-v1'
    assert gate['official_accepted'] is True and gate['release_majority_score'] is True
    assert gate['competition_id'] == 532512 and gate['season_id'] == 1862
    assert gate['method_dev_sha256'] == METHOD and gate['B_input_sha256'] == B_INPUT
    assert isinstance(gate['platform_submission_id'], (str, int)) and str(gate['platform_submission_id']).strip()
    accepted = datetime.fromisoformat(gate['accepted_at_utc'].replace('Z', '+00:00'))
    assert accepted.tzinfo is not None and accepted <= datetime.now(timezone.utc)
    checked(gate['official_receipt']); checked(gate['uploaded_prediction'])
    assert Path(gate['official_receipt']['path']).resolve() != Path(path).resolve()
    return actual
