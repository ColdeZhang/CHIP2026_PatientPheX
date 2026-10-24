"""Freeze authorized complete58 only after the one exact F0 probe passes."""
from pathlib import Path
import argparse
import ast
import datetime
import gzip
import hashlib
import io
import json
import re
import shutil
import tarfile


def ref(path):
    data = Path(path).read_bytes()
    return {'path': str(Path(path).resolve()), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f0-root', type=Path, required=True)
    parser.add_argument('--repo-full58', type=Path, required=True)
    args = parser.parse_args()
    root = args.f0_root.resolve()
    old = root.parent / 'full80-v1'
    probe = root / 'probe-v1'
    terminal = probe / 'terminal'
    cpu = json.loads((root / 'input-cpu/result.json').read_text())
    pplan = json.loads((probe / 'plan.json').read_text())
    summary = json.loads((terminal / 'summary.json').read_text())
    wrapper = json.loads((terminal / 'wrapper-complete.json').read_text())
    release = json.loads((terminal / 'gpu-release.json').read_text())
    assets = json.loads((terminal / 'actual-input-and-assets.json').read_text())
    assert wrapper['returncode'] == 0 and release['target_compute_rows'] == []
    assert len(release['target_csv']) == 1 and release['target_csv'][0].split(',')[1].strip() == pplan['gpu_uuid']
    assert summary['status'] == 'PASS_NATIVE_LONGEST_STRICT_COMPLETE'
    assert summary['strict_complete'] and summary['natural_eos'] and not summary['hit_token_limit']
    assert summary['generation_seconds'] <= 30 and summary['max_allocated_bytes'] <= 60_000_000_000
    assert summary['plan_sha256'] == ref(probe / 'plan.json')['sha256']
    assert summary['source_sha256'] == ref(probe / 'runner.py')['sha256']
    assert summary['index'] == pplan['probe_index'] == min(cpu['expanded']['longest_indices'])
    assert summary['input_tokens'] == cpu['expanded']['maximum']
    assert assets['adapter_files'] == {name: {'bytes': value['bytes'], 'sha256': value['sha256']} for name, value in json.loads((root / 'source-plan.json').read_text())['original_SFT48_adapter']['files'].items()}
    target = root / 'full58-v1'
    target.mkdir(exist_ok=True)
    assert not (target / 'plan.json').exists() and not (target / 'start-intent.json').exists()
    for name in ('legacy_runner.py', 'model-source-manifest.json'):
        shutil.copyfile(probe / name, target / name)
    shutil.copyfile(probe / 'runner.py', target / 'probe_runner.py')
    shutil.copyfile(root / 'expanded58.jsonl', target / 'expanded58.jsonl')
    shutil.copyfile(root / 'input-cpu/row-audit.json', target / 'row-audit.json')
    for before, after in [('raw-response.json', 'cache-probe-raw.json'), ('generated-ids.json', 'cache-probe-ids.json'), ('format-result.json', 'cache-probe-format.json'), ('summary.json', 'cache-probe-summary.json')]:
        shutil.copyfile(terminal / before, target / after)
    original = (old / 'infer_full80.py').read_text()
    source = original.replace('complete80 SFT32', 'complete58 original-SFT48 F0').replace('79 fresh', '57 fresh')
    source = source.replace("plan = json.loads(Path('plan.json').read_text())", "plan = json.loads(Path('plan.json').read_text())\n    cached_index = plan['cached_probe_index']")
    source = source.replace("assert plan['input_limit'] == 14400 and plan['generation_limit'] == 1600", "assert plan['input_limit'] == ((plan['expected_input_max'] + 511) // 512) * 512\n    assert plan['expected_input_max'] <= 32768 and plan['generation_limit'] == 1600")
    for before, after in {'80': '58', '79': '57', '587': '415', '19': 'cached_index', '670122': "plan['expected_input_tokens']", '14390': "plan['expected_input_max']", '15990': "(plan['input_limit'] + plan['generation_limit'])"}.items():
        source = re.sub(r'\b' + before + r'\b', after, source)
    source = source.replace('expanded80.jsonl', 'expanded58.jsonl').replace('all80_actual_ids_equal_prior_native_CPU', 'all58_actual_ids_equal_prior_native_CPU').replace('COMPLETE_FIXED80_SFT32_FULL_REPORT_NO_QUALITY', 'COMPLETE_FIXED58_SFT48_FULL_REPORT_NO_QUALITY')
    compile(source, 'infer_full58.py', 'exec')
    (target / 'infer_full58.py').write_text(source)
    a, b = ast.parse(original), ast.parse(source)
    calls = lambda tree, name: [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name]
    assert calls(a, 'generate') == calls(b, 'generate') and calls(a, 'from_pretrained') == calls(b, 'from_pretrained')
    helpers = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert all(helpers(a)[name] == helpers(b)[name] for name in ('jsonl', 'copy_exact'))
    launch = (old / 'launch.py').read_text().replace('infer_full80.py', 'infer_full58.py')
    operations = (old / 'operations.py').read_text().replace('infer_full80.py', 'infer_full58.py').replace('full80', 'full58').replace('FULL80', 'FULL58')
    for name, value in [('launch.py', launch), ('operations.py', operations)]:
        compile(value, name, 'exec')
        (target / name).write_text(value)
    decision = json.loads((root / 'execution-policy.json').read_text())
    decision['actual_conditional_probe'] = {'summary': ref(terminal / 'summary.json'), 'wrapper': ref(terminal / 'wrapper-complete.json'), 'release': ref(terminal / 'gpu-release.json'), 'passed': True, 'no_quality': True}
    save(target / 'decision.json', decision)
    keys = ['mount_id', 'gpu_uuid', 'python', 'model', 'adapter', 'adapter_files', 'env', 'generation', 'baseline_old_SFT48_reference', 'actual_CPU']
    plan = {key: pplan[key] for key in keys}
    plan.update(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), schema='qwen35-original-SFT48-F0-append-full-report-full58-v1', remote='ontology-round2/qwen35-sft48-full-report-f0-full58-v1', cap_seconds=1200, term_grace_seconds=30, input_limit=pplan['input_limit'], generation_limit=1600, cached_original_indices=[pplan['probe_index']], cached_probe_index=pplan['probe_index'], fresh_rows=57, expected_rows=58, expected_input_tokens=cpu['expanded']['total'], expected_input_max=cpu['expanded']['maximum'], output_order='ascending(input_tokens,row_id)')
    plan['cached_probe_runtime'] = {key: assets[key] for key in ['torch', 'transformers', 'peft', 'tokenizers', 'actual_tokenizer_class', 'matmul_TF32', 'cudnn_TF32']}
    plan['probe_plan'] = ref(probe / 'plan.json')
    plan['probe_terminal_manifest'] = ref(probe / 'terminal-manifest.json')
    plan['probe_summary'] = ref(terminal / 'summary.json')
    plan['CPU_row_audit'] = ref(root / 'input-cpu/row-audit.json')
    plan['decision'] = ref(target / 'decision.json')
    names = ['infer_full58.py', 'launch.py', 'probe_runner.py', 'legacy_runner.py', 'model-source-manifest.json', 'expanded58.jsonl', 'row-audit.json', 'cache-probe-raw.json', 'cache-probe-ids.json', 'cache-probe-format.json', 'cache-probe-summary.json', 'decision.json']
    plan['files'] = {name: {key: ref(target / name)[key] for key in ('bytes', 'sha256')} for name in names}
    save(target / 'plan.json', plan)
    names.append('plan.json')
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode='w') as archive:
        for name in names:
            data = (target / name).read_bytes()
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(data), 0o600, 0
            archive.addfile(item, io.BytesIO(data))
    (target / 'deployment.tar.gz').write_bytes(gzip.compress(raw.getvalue(), compresslevel=9, mtime=0))
    save(target / 'deployment-manifest.json', {'files': {name: ref(target / name) for name in names}, 'archive': ref(target / 'deployment.tar.gz'), 'plan': ref(target / 'plan.json')})
    save(target / 'source-checks.json', {'status': 'FIXED58_F0_IDENTITY_ONLY_NO_GENERATION_OR_CACHE_POLICY_CHANGE', 'prior_runner': ref(old / 'infer_full80.py'), 'native_generation_and_load_calls_AST_equal': True, 'durable_copy_and_jsonl_helpers_AST_equal': True, 'conditional_probe_satisfied': True, 'model_calls': 0, 'quality_calls': 0})
    save(target / 'entry.json', {'status': 'CONDITIONALLY_AUTHORIZED_FIXED58_SOURCE_READY_NOT_STARTED', 'plan': ref(target / 'plan.json'), 'runner': ref(target / 'infer_full58.py'), 'launcher': ref(target / 'launch.py'), 'operations': ref(target / 'operations.py'), 'source_checks': ref(target / 'source-checks.json'), 'archive': ref(target / 'deployment.tar.gz'), 'manifest': ref(target / 'deployment-manifest.json'), 'limits': 'One1200+TERM30 exact probe1 plus57fresh. Predeclared two endpoints only after complete58. No automatic originaldev/official migration.'})
    args.repo_full58.mkdir(parents=True, exist_ok=True)
    for name in ['infer_full58.py', 'launch.py', 'operations.py', 'plan.json', 'decision.json', 'entry.json', 'source-checks.json', 'deployment-manifest.json']:
        shutil.copyfile(target / name, args.repo_full58 / name)
    print(json.dumps({'entry': ref(target / 'entry.json'), 'plan': ref(target / 'plan.json'), 'runner': ref(target / 'infer_full58.py'), 'launcher': ref(target / 'launch.py'), 'operations': ref(target / 'operations.py'), 'archive': ref(target / 'deployment.tar.gz')}))


if __name__ == '__main__':
    main()
