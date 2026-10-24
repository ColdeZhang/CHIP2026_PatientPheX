"""Fixed two-endpoint F1 control. Gold is inaccessible before the score worker gate."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ARMS = ['primary', 'new-only']
METRICS = ('menF1', 'docF1', 'micF1', 'macF1', 'Score')
KEEP = 'choose highest Score of the two fixed endpoints, tie primary; retain only strictly above strong N; macro disclosed without veto'
BASE_SCORE = 0.676546354662594
GOLD = dict(path='/workspace/patientphex-data/round2-inner4/fold-1/validation16.gold.jsonl',
            bytes=525710, sha256='665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3')
PINS = {
    'baseline': 'f33a30f1f3c505022dbcf846a19e6496b3810627808e97cde595a9ed53d17866',
    'baseline_metrics': '875292c3eb226b484ff86309cb92e5d28dca6f3857767481d5cb56cf9c68ee86',
    'F1_input': '962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415',
    'byte_helper': 'c713d2f5d5b2768cfb80ce4c17f920c6228dc3abb833080c972255d6e34c6e06',
    'strict': '31d380f582a6ab83e0c73bddbac983c5cf038b250dd2b29355c14f11d8eaebf5',
    'evaluator': 'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781',
    'validator': 'b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f',
    'hpo': 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',
    'old_prompts': '2a95bad1683e3a622650346128582a7cff91c872e1f64890b5e380680a5de085',
}


def now(): return datetime.now(timezone.utc).isoformat()


def ref(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def checked(item):
    assert isinstance(item, dict) and ref(item['path']) == item, item
    return Path(item['path'])


def read(path): return json.loads(Path(path).read_text())


def publish(path, raw):
    path = Path(path); temporary = path.with_name(path.name + '.pending')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()
    parent = path.resolve().parent
    while True:
        fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        if parent.parent == parent: break
        parent = parent.parent


def save(path, value): publish(path, (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def jsonl(path): return [json.loads(line) for line in Path(path).read_text().splitlines()]


def rows(path):
    lines = Path(path).read_bytes().decode().splitlines(keepends=True)
    docs = [json.loads(line) for line in lines]; ids = [str(d['pmc_id']) for d in docs]
    assert len(ids) == len(set(ids))
    return lines, docs, ids


def load_module(name, item):
    spec = importlib.util.spec_from_file_location(name, checked(item))
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def gold_guard(plan):
    target = os.path.realpath(plan['gold']['path']); state = {'enabled': False}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) == target and not state['enabled']:
                raise PermissionError('Gold open/hash forbidden until both complete endpoints are frozen')
    sys.addaudithook(audit); return state


def load_plan(path):
    assert __debug__, 'Assertions must remain enabled'
    plan = read(path); guard = gold_guard(plan)
    assert plan['schema'] == 'grounded-F1-two-fixed-endpoints-v1'
    assert plan['state'] == 'BOUND_COMPLETE39_AND_NORMALIZATIONS', 'Unbound template cannot run'
    assert plan['arms'] == ARMS and plan['keep_rule'] == KEEP and plan['baseline_score'] == BASE_SCORE
    assert plan['gold'] == GOLD  # metadata only; never stat/open/hash here
    assert plan['expected'] == dict(documents=16, patients=39, entities=1189, generated_rows=39,
                                    old_prompt_rows=80, old_candidate_codes=587, old_candidate_pairs=1557)
    assert plan['score_budget_seconds'] == 30 and plan['term_grace_seconds'] == 1
    assert plan['new_evaluator_calls'] == 2 and plan['old_evaluator_calls'] == 0
    for name, item in plan['sources'].items():
        assert Path(item['path']).name == name; checked(item)
    assert set(plan['sources']) == {'support.py', 'assemble.py', 'score.py', 'contract_cases.py'}
    assert plan['sources']['support.py'] == ref(__file__)
    for name, item in plan['references'].items():
        checked(item)
        if name in PINS: assert item['sha256'] == PINS[name]
    assert set(PINS) <= set(plan['references'])
    baseline = read(plan['references']['baseline_metrics']['path'])
    assert baseline['metrics']['Score'] == BASE_SCORE
    assert baseline['evaluator_version'] == 'patientphex-public-approx-v2'
    assert baseline['configuration']['body_scope'] == 'all'
    assert baseline['configuration']['unlinked_document_mode'] == 'text'
    # Original public-v2 metrics lacks hashes: bind its actual frozen provenance chain.
    summary = read(plan['references']['baseline_quality_summary']['path'])
    frozen = read(plan['references']['baseline_quality_freeze']['path'])
    bound = read(plan['references']['baseline_evaluation_bound']['path'])
    endpoint = summary['endpoints']['diagnostic-fulltext-alone']
    assert endpoint['metrics_file'] == plan['references']['baseline_metrics']
    assert endpoint['metrics'] == baseline['metrics']
    assert summary['paired_freeze'] == plan['references']['baseline_quality_freeze']
    assert frozen['endpoints']['diagnostic-fulltext-alone']['prediction'] == plan['references']['baseline']
    assert frozen['plan'] == plan['references']['baseline_evaluation_bound']
    assert bound['gold_path'] == GOLD['path'] and bound['historical_F1_gold_sha256'] == GOLD['sha256']
    assert bound['references']['input'] == plan['references']['F1_input']
    assert (baseline['documents'], baseline['patients']) == (16, 39)
    assert isinstance(plan['actual'], dict) and set(plan['actual']) == {
        'collection', 'extraction_plan', 'extraction_manifest', 'quotes', 'decisions',
        'normalization_manifest', 'normalizations', 'semantic_evidence'}
    for item in plan['actual'].values(): checked(item)
    return plan, guard


def byte_helpers(item):
    assert item['sha256'] == PINS['byte_helper']
    tree = ast.parse(checked(item).read_text())
    names = ['field_span', 'array_spans', 'phenotype_spans', 'mask_values']
    nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name) for name in names]
    namespace = {'json': json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), item['path'], 'exec'), namespace)
    return namespace['phenotype_spans'], namespace['mask_values']


def replace_phenotypes(line, values, spans, mask):
    doc = json.loads(line); positions = spans(line)
    assert len(positions) == len(doc['association'])
    result = line
    for item, (begin, end) in reversed(list(zip(doc['association'], positions))):
        selected = values[item['patient_id']]
        if selected != item['phenotype']:
            result = result[:begin] + json.dumps(selected, ensure_ascii=False) + result[end:]
    assert mask(result) == mask(line), 'Every byte outside phenotype arrays must remain unchanged'
    return result


def canonical_values(values, hpo):
    """Stable canonical atomic keys; NIL strings containing semicolons stay whole."""
    result = []; seen = set()
    for value in values:
        for key in hpo.relation_keys(value):
            if key not in seen:
                seen.add(key); result.append(key[1])
    return result


def relation_keys(docs, hpo):
    result = set()
    for doc in docs:
        pids = set()
        for association in doc['association']:
            pid = association['patient_id']; assert pid not in pids; pids.add(pid)
            for value in association['phenotype']:
                result.update((str(doc['pmc_id']), pid, *key) for key in hpo.relation_keys(value))
    return result


def endpoint_choice(metrics):
    assert set(metrics) == set(ARMS)
    # Python max is stable: primary wins an exact total-Score tie.
    winner = max(ARMS, key=lambda arm: metrics[arm]['Score'])
    return winner, metrics[winner]['Score'] > BASE_SCORE


def validate_freeze(plan_path, plan):
    path = Path(plan['output_directory']) / 'assembled/common-freeze.json'; frozen = read(path)
    assert frozen['plan'] == ref(plan_path) and frozen['sources'] == plan['sources']
    assert frozen['actual'] == plan['actual'] and list(frozen['predictions']) == ARMS
    assert frozen['both_complete_endpoints'] and frozen['gold_read_or_hashed'] is False
    assert frozen['new_evaluator_calls_before_freeze'] == 0
    for item in frozen['predictions'].values(): checked(item)
    contract = read(checked(frozen['contract'])); checked(frozen['contributions']); checked(frozen['normalization_consumption'])
    for item in frozen['validations'].values(): assert read(checked(item))['valid']
    assert contract['documents'] == 16 and contract['patients'] == 39 and contract['entities'] == 1189
    assert contract['all39_raw_covered'] and contract['all_nonphenotype_bytes_exact']
    assert contract['primary_is_N_union_new'] and contract['diagnostic_is_new_only']
    assert contract['gold_read_or_hashed'] is False
    return path, frozen
