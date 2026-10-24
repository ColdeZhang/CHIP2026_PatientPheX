"""Thin F0 metadata/source check; reuse native receipts, no model/tokenizer/gold."""
import ast
import gzip
import hashlib
import json
import os
import sys
import time
from common import B, G, ROUTE, S, checked, read, ref, save, verify_sources


def main():
    began = time.monotonic()
    forbidden = os.path.realpath('/workspace/patientphex-data/round2-inner4/fold-0/validation16.gold.jsonl')
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)) and os.path.realpath(os.fsdecode(args[0])) == forbidden:
            raise PermissionError('No held gold in F0 source preparation')
    sys.addaudithook(audit)
    manifest = verify_sources()
    plan = read(G / 'f0/infer-plan.json')
    refs = {n: plan['payload_sources'][n] for n in ('held-prompts.jsonl', 'held-actual-ids.json.gz', 'held-audit.json', 'native-CPU-result.json')}
    for item in refs.values(): checked(item)
    prompts = [json.loads(x) for x in checked(refs['held-prompts.jsonl']).read_text().splitlines()]
    encoded = json.loads(gzip.decompress(checked(refs['held-actual-ids.json.gz']).read_bytes()))
    audit_rows = read(checked(refs['held-audit.json']))
    ids = {r['row_id']: r['input_ids'] for r in encoded}
    rows = {r['row_id']: r for r in prompts}
    assert len(ids) == len(rows) == len(audit_rows) == 47
    for row in audit_rows:
        tokens = ids[row['row_id']]
        assert len(tokens) == row['input_tokens']
        assert hashlib.sha256(json.dumps(tokens).encode()).hexdigest() == row['input_ids_sha256']
    native = read(refs['native-CPU-result.json']['path'])
    assert native['all47_local_native_actual_IDs_rendering_decode_exact']
    assert native['encoded']['sha256'] == refs['held-actual-ids.json.gz']['sha256']
    assert sum(map(len, ids.values())) == native['input_tokens'] == 289420
    assert max(map(len, ids.values())) == native['max_input_tokens'] == 9935
    pure = read(G / 'f0/pure-plan.json')['references']['F0_input']
    docs = [json.loads(x) for x in checked(pure).read_text().splitlines()]
    pairs = {(str(d['pmc_id']), p['patient_id']) for d in docs for p in d['patient']}
    assert len(docs) == 16 and len(pairs) == 47
    assert pairs == {(str(r['pmc_id']), r['target_patient_id']) for r in prompts}
    assert all('answer' not in r and 'labels' not in r for r in prompts)
    # Existing audit contains identity metadata only; do not open training targets.
    trainplan = read(ROUTE / 'training/fresh_v1/formal-plan.json')
    training = read(checked(trainplan['payload_sources']['train-audit.json']))
    assert len(training) == 87
    train_docs = {str(r['pmc_id']) for r in training}
    assert len(train_docs) == 32 and not train_docs & {str(d['pmc_id']) for d in docs}
    from bind import validate_terminal
    validate_terminal(B / 'formal-fresh-v1', trainplan)
    def generate(path):
        tree = ast.parse(path.read_text())
        return [ast.dump(n, include_attributes=False) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'generate']
    assert generate(S / 'run.py') == generate(ROUTE / 'inference_helper/run.py') and len(generate(S / 'run.py')) == 1
    assert (S / 'raw_store.py').read_bytes() == (ROUTE / 'inference_helper/raw_store.py').read_bytes()
    compiled = []
    for path in sorted(S.rglob('*.py')):
        tree = ast.parse(path.read_text()); compile(tree, str(path), 'exec'); compiled.append(str(path.relative_to(S)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and any(isinstance(t, ast.Name) and t.id.endswith('_CODE') for t in node.targets):
                compile(node.value.value, str(path) + ':embedded', 'exec')
    from admission import validate, admit
    p = dict(budget_seconds=2100, authorized_budget_seconds=2100, grace_seconds=30,
             admission=dict(schema='training-seed-F0-admission-v1', bound_at_unix=1000,
                 launch_not_after_unix=1300, deadline_TERM_unix=3100, deadline_KILL_unix=3130))
    validate(p); admit(p, 'synthetic_entry', 1200)
    try: admit(p, 'expired_entry', 1301)
    except AssertionError: pass
    else: raise AssertionError('Expired launch admitted')
    complete = read(B / 'infer-F1-v1/terminal/output/complete.json')
    wrapper = read(B / 'infer-F1-v1/terminal/wrapper-complete.json')
    f1actual = wrapper['total_controller_seconds']
    cost = dict(F1_actual_controller_seconds=f1actual,
        patient_ratio_proxy_seconds=f1actual * 47 / 39,
        input_ratio_proxy_plus_120_seconds=f1actual * 289420 / 203588 + 120,
        budget_seconds=2100, grace_seconds=30, estimates_not_guarantees=True)
    assert cost['input_ratio_proxy_plus_120_seconds'] < 2100
    result = dict(status='PASS_COMPLETE_F0_SOURCE_METADATA_CONTRACT', source=ref(__file__),
        source_manifest=ref(S / 'source-manifest.json'), refs=refs, rows=47, documents=16,
        input_tokens=289420, max_input_tokens=9935, all_input_ID_hashes_exact=True,
        original_native_receipt_reused_no_retokenization=True, train32_disjoint=True,
        new_adapter='c3dc7de5940c67bf82ff9d0fa9a22d5ebc3afe11008f6a69bdef7e25d924fa43',
        same_generate_AST_and_raw_store=True, compiled_sources=compiled,
        actual_F1_allocated_bytes=complete['max_gpu_allocated_bytes'], reserved_not_recorded=True,
        A100_selection='Resource queue choice, not proof A30 unsuitable; same cap1600, F0 max9935 < F1 max10080; root allows bounded A30 attempt if an actual unconflicted slot is later assigned.',
        cost=cost, NN=0, tokenizer_calls=0, fits=0, quality_calls=0, seconds=time.monotonic()-began)
    save(S / 'cpu-contract.json', result)
    print(json.dumps(ref(S / 'cpu-contract.json')))


if __name__ == '__main__': main()
