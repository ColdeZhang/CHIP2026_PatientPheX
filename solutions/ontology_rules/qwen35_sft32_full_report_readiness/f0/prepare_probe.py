"""Freeze one F0 longest probe only after the original58 native CPU census."""
from pathlib import Path
import argparse
import ast
import datetime
import gzip
import hashlib
import io
import json
import shutil
import tarfile


def ref(path):
    data = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f0-root', type=Path, required=True)
    parser.add_argument('--repo-probe', type=Path, required=True)
    args = parser.parse_args()
    root = args.f0_root.resolve()
    cpu = root / 'input-cpu'
    result = json.loads((cpu / 'result.json').read_text())
    complete = json.loads((cpu / 'CPU-complete.json').read_text())
    audit = json.loads((cpu / 'row-audit.json').read_text())
    assert complete['returncode'] == 0 and result['probe_input_gate_passed']
    assert result['rows'] == len(audit) == 58 and result['candidates'] == 415 and result['patients'] == 47
    assert result['source_sha256'] == ref(cpu / 'inspect_inputs.py')['sha256']
    assert result['plan_sha256'] == ref(cpu / 'plan.json')['sha256']
    assert result['row_audit'] == {k: ref(cpu / 'row-audit.json')[k] for k in ('bytes', 'sha256')}
    maximum = result['expanded']['maximum']
    index = min(result['expanded']['longest_indices'])
    limit = ((maximum + 511) // 512) * 512
    assert maximum <= 32768 and limit == result['recommended_input_limit']
    assert limit + 1600 <= result['model_context_tokens']
    rows = [json.loads(line) for line in (root / 'expanded58.jsonl').read_text().splitlines()]
    row = rows[index]
    jsha = lambda obj: hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert len(rows) == 58 and jsha(row) == audit[index]['expanded_row_sha256']
    old = root.parent / 'probe-v1'
    target = root / 'probe-v1'
    target.mkdir(exist_ok=False)
    for name in ('legacy_runner.py', 'model-source-manifest.json', 'launch.py'):
        shutil.copyfile(old / name, target / name)
    original = (old / 'runner.py').read_text()
    runner = original.replace('One fixed SFT32 full-report input', 'One fixed original-SFT48 F0 full-report input')
    runner = replace_once(runner, "assert plan['input_limit'] == 14400 and plan['generation_limit'] == 1600", "assert plan['input_limit'] == ((plan['row_audit']['expanded']['tokens'] + 511) // 512) * 512\n    assert plan['row_audit']['expanded']['tokens'] <= 32768 and plan['generation_limit'] == 1600")
    runner = replace_once(runner, "assert plan['probe_index'] == 19 and plan['cap_seconds'] == 240", "assert 0 <= plan['probe_index'] < 58 and plan['cap_seconds'] == 240")
    runner = replace_once(runner, "assert len(ids) == expected['tokens'] == 14390 <= plan['input_limit']", "assert len(ids) == expected['tokens'] <= plan['input_limit']")
    assert runner.count("'index': 19") == 2
    runner = runner.replace("'index': 19", "'index': plan['probe_index']")
    compile(runner, 'runner.py', 'exec')
    (target / 'runner.py').write_text(runner)
    before, after = ast.parse(original), ast.parse(runner)
    functions = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert all(functions(before)[n] == functions(after)[n] for n in ('jsha', 'save', 'unique_object', 'parse_complete'))
    calls = lambda tree, name: [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name]
    assert calls(before, 'generate') == calls(after, 'generate')
    assert calls(before, 'from_pretrained') == calls(after, 'from_pretrained')
    save(target / 'source-checks.json', {'status': 'F0_ONLY_IDENTITY_LENGTH_ADAPTER_PLAN_CHANGE', 'old_runner': ref(old / 'runner.py'), 'native_generation_and_load_calls_AST_equal': True, 'four_raw_parser_helpers_AST_equal': True, 'no_model_gold_or_quality': True})
    (target / 'probe.jsonl').write_text(json.dumps(row, ensure_ascii=False) + '\n')
    shutil.copyfile(root / 'execution-policy.json', target / 'decision.json')
    plan = json.loads((old / 'plan.json').read_text())
    plan['frozen_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan['schema'] = 'qwen35-original-SFT48-F0-append-full-report-longest-v1'
    plan['remote'] = 'ontology-round2/qwen35-sft48-full-report-f0-probe-v1'
    plan['adapter'] = '/dev/shm/patientphex-sequence-sft48-reverse-f0-1045'
    source = json.loads((root / 'source-plan.json').read_text())
    plan['adapter_files'] = {name: value['sha256'] for name, value in source['original_SFT48_adapter']['files'].items()}
    plan['input_limit'], plan['probe_index'], plan['row_audit'] = limit, index, audit[index]
    plan.pop('baseline_old_SFT32_reference')
    plan['baseline_old_SFT48_reference'] = source['references']
    plan['actual_CPU'] = ref(cpu / 'result.json')
    plan['CPU_row_audit'] = ref(cpu / 'row-audit.json')
    plan['CPU_complete'] = ref(cpu / 'CPU-complete.json')
    plan['recovery_readonly'] = ref(root / 'asset-readiness/target800.json')
    plan['longest_tie_policy'] = 'Smallest original index among exactly maximal actual input lengths.'
    names = ['legacy_runner.py', 'launch.py', 'runner.py', 'probe.jsonl', 'source-checks.json', 'model-source-manifest.json', 'decision.json']
    plan['files'] = {name: {k: ref(target / name)[k] for k in ('bytes', 'sha256')} for name in names}
    save(target / 'plan.json', plan)
    operations = (old / 'operations.py').read_text()
    old_guard = "assert json.loads((B / 'deployment-result.json').read_text())['status'] == 'ACTUAL_ALL_SOURCE_INPUT_SHA_COMPILE_FSYNC'"
    operations = replace_once(operations, old_guard, old_guard + "\n    assert json.loads((B / 'deployment-result.json').read_text())['plan_sha256'] == hashlib.sha256((B / 'plan.json').read_bytes()).hexdigest()")
    compile(operations, 'operations.py', 'exec')
    (target / 'operations.py').write_text(operations)
    names.append('plan.json')
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as archive:
        for name in names:
            data = (target / name).read_bytes()
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(data), 0o600, 0
            archive.addfile(item, io.BytesIO(data))
    (target / 'deployment.tar.gz').write_bytes(gzip.compress(raw.getvalue(), compresslevel=9, mtime=0))
    assert (target / 'deployment.tar.gz').stat().st_size <= 65536
    save(target / 'deployment-manifest.json', {'files': {name: ref(target / name) for name in names}, 'archive': ref(target / 'deployment.tar.gz'), 'plan': ref(target / 'plan.json')})
    save(target / 'entry.json', {'status': 'F0_LONGEST_SOURCE_READY_NOT_STARTED', 'plan': ref(target / 'plan.json'), 'runner': ref(target / 'runner.py'), 'launcher': ref(target / 'launch.py'), 'operations': ref(target / 'operations.py'), 'archive': ref(target / 'deployment.tar.gz'), 'manifest': ref(target / 'deployment-manifest.json'), 'actual_CPU': ref(cpu / 'result.json'), 'actual_recovery_readonly': ref(root / 'asset-readiness/target800.json'), 'limits': 'One 240+TERM30 engineering probe, no gold/quality. Full58 conditional on frozen root policy.'})
    args.repo_probe.mkdir(parents=True, exist_ok=False)
    for name in ['runner.py', 'launch.py', 'operations.py', 'plan.json', 'decision.json', 'entry.json', 'source-checks.json', 'deployment-manifest.json']:
        shutil.copyfile(target / name, args.repo_probe / name)
    print(json.dumps({'entry': ref(target / 'entry.json'), 'plan': ref(target / 'plan.json'), 'runner': ref(target / 'runner.py'), 'index': index, 'maximum': maximum, 'input_limit': limit, 'archive_bytes': (target / 'deployment.tar.gz').stat().st_size}))


if __name__ == '__main__':
    main()
