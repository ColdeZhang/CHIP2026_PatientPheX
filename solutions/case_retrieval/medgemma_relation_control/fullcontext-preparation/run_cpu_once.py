#!/usr/bin/env python3
"""One separately authorized .100 tokenizer CPU job; ambiguous launch is never retried."""
from pathlib import Path
import base64
import datetime
import hashlib
import json
import shlex
import sys
import time

sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
D = Path(__file__).resolve().parent
MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
REMOTE = 'research/medgemma-relation-control/fullcontext-preparation-v1'
OUTPUT = REMOTE + '/native-cpu'
MODEL = 'research/medical-small-model-readiness/models/medgemma-1.5-4b-it-9bca96ca3e30'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (D / name).open('x') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        handle.flush()
        import os
        os.fsync(handle.fileno())


def main():
    assert not (D / 'cpu-launch-plan.json').exists(), 'Existing preparation/intent: inspect it, never blindly relaunch'
    files = {name: D / name for name in ['native_audit.py', 'construct.py', 'legacy_full_context.py', 'medgemma_interface.py']}
    files['expected-files.json'] = D.parents[1] / 'medical-small-model-readiness/assets/expected-files.json'
    bundle = D / 'budget-inputs.json.gz'
    refs = {name: {'local': str(path), 'remote': REMOTE + '/' + name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
            for name, path in files.items()}
    refs['budget-inputs.json.gz'] = {'local': str(bundle), 'remote': REMOTE + '/budget-inputs.json.gz',
                                    'bytes': bundle.stat().st_size, 'sha256': sha(bundle)}
    environment = {'CUDA_VISIBLE_DEVICES': '',
                   'PYTHONPATH': '/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/qwen35-preparation/a30-fullcontext/runtime:/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/round2/runtime',
                   'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                   'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
    argv = ['/usr/bin/timeout', '--signal=KILL', '60', '/home/zhangyuheng/projects/ScratchVL/.venv/bin/python',
            '-u', REMOTE + '/native_audit.py', '--bundle', refs['budget-inputs.json.gz']['remote'],
            '--bundle-sha256', refs['budget-inputs.json.gz']['sha256'], '--model', MODEL,
            '--source-manifest', REMOTE + '/expected-files.json', '--output-directory', OUTPUT]
    plan = {'status': 'single_native_CPU_tokenizer_audit', 'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'mount_id': MOUNT, 'namespace': REMOTE, 'files': refs, 'argv': argv, 'env': environment,
            'budget_seconds': 60, 'grace_seconds': 0, 'no_GPU_model_weights_training_inference_quality': True,
            'launcher_sha256': sha(__file__), 'preparation_sha256': sha(D / 'preparation.json')}
    save('cpu-launch-plan.json', plan)
    client, job = Client(), None
    def call(name, arguments):
        result = client.request('tools/call', {'name': name, 'arguments': {'mount_id': MOUNT, **arguments}})
        assert not result.get('isError'), 'MCP returned tool error; preserve current state, no automatic retry'
        return result.get('structuredContent') or json.loads(result['content'][0]['text'])
    def command(code, seconds=20):
        result = call('run_command', {'command': 'python3 -c ' + shlex.quote(code), 'cwd': '.', 'timeout_seconds': seconds})
        assert result['exit_code'] == 0, result
        return result
    try:
        client.initialize()
        save('remote-preflight.json', command('from pathlib import Path;p=Path(' + repr(REMOTE) + ');assert not p.exists();p.mkdir(parents=True,mode=0o700);print("unique CPU-only namespace created")'))
        upload_receipts = []
        for name, path in files.items():
            record = refs[name]
            receipt = call('write_file', {'path': record['remote'], 'content': base64.b64encode(path.read_bytes()).decode(), 'encoding': 'base64', 'create_parents': False})
            upload_receipts.append({'name': name, 'expected_sha256': record['sha256'], 'receipt': receipt})
        parts = []
        payload = bundle.read_bytes()
        for index, offset in enumerate(range(0, len(payload), 256 * 1024)):
            data = payload[offset:offset + 256 * 1024]
            name = REMOTE + f'/budget-inputs.part-{index:03d}'
            record = {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            receipt = call('write_file', {'path': name, 'content': base64.b64encode(data).decode(), 'encoding': 'base64', 'create_parents': False})
            upload_receipts.append({'part': record, 'receipt': receipt})
            parts.append(record)
        save('upload-receipts.json', upload_receipts)
        code = ('from pathlib import Path;import hashlib,json;parts=' + repr(parts)
                + ';assert all(Path(r["path"]).stat().st_size==r["bytes"] and hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest()==r["sha256"] for r in parts);'
                + 'p=Path(' + repr(refs['budget-inputs.json.gz']['remote']) + ');assert not p.exists();'
                + 'p.write_bytes(b"".join(Path(r["path"]).read_bytes() for r in parts));refs=' + repr(refs)
                + ';assert all(Path(r["remote"]).stat().st_size==r["bytes"] and hashlib.sha256(Path(r["remote"]).read_bytes()).hexdigest()==r["sha256"] for r in refs.values());print(json.dumps(dict(all_sources_and_bundle_actual_SHA_match=True,files=len(refs))))')
        save('remote-source-verification.json', command(code))
        save('execution-intent.json', {'at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                       'plan_sha256': sha(D / 'cpu-launch-plan.json'), 'argv': argv,
                                       'budget_seconds': 60, 'no_relaunch_if_network_state_unknown': True})
        # The only CPU process launch. Never retried even if this call is ambiguous.
        job = call('start_process', {'command': shlex.join(argv), 'cwd': '.', 'env': environment})
        save('known-job.json', job)
        print(json.dumps({'known_CPU_job': job['id'], 'budget_seconds': 60}), flush=True)
        cursors = {'stdout': 0, 'stderr': 0}
        began = time.monotonic()
        while True:
            running = False
            for stream in cursors:
                response = call('read_process_output', {'process_id': job['id'], 'stream': stream, 'cursor': cursors[stream], 'limit': 20000})
                cursors[stream] = response['next_cursor']
                running |= response['running']
                if response['content']:
                    with (D / (stream + '.log')).open('a') as handle:
                        handle.write(response['content'])
                    if stream == 'stdout':
                        print(response['content'], end='', flush=True)
            if not running:
                break
            assert time.monotonic() - began < 150, 'Known CPU job monitor uncertain; inspect same job, do not launch replacement'
            time.sleep(3)
        save('process-terminal.json', {'job': job, 'running': False, 'cursors': cursors,
                                       'observed_elapsed_seconds': time.monotonic() - began,
                                       'remote_exit_code_available': False})
        listing = command('from pathlib import Path;import hashlib,json;p=Path(' + repr(OUTPUT) + ');print(json.dumps({f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in p.glob("*.json") if f.stat().st_size<1000000}))')
        records = json.loads(listing['stdout'])
        save('remote-output-manifest.json', records)
        local = D / 'actual-cpu'
        local.mkdir(exist_ok=False)
        for name, record in records.items():
            assert Path(name).name == name
            data = bytearray()
            for offset in range(0, record['bytes'], 256 * 1024):
                response = call('read_file', {'path': OUTPUT + '/' + name, 'offset': offset, 'limit': min(256 * 1024, record['bytes'] - offset)})
                data.extend(base64.b64decode(response['content']) if response.get('encoding') == 'base64' else response['content'].encode())
            assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
            (local / name).write_bytes(data)
        assert 'summary.json' in records and 'failure.json' not in records, 'CPU audit failed/incomplete; preserve evidence, no retry'
        summary = json.loads((local / 'summary.json').read_text())
        assert summary['status'] == 'complete_native_CPU_fullcontext_budget_only' and not summary['cuda_initialized']
        assert summary['source_sha256'] == refs['native_audit.py']['sha256']
        save('terminal-receipt.json', {'status': 'complete_CPU_only', 'job': job, 'summary': records['summary.json'],
                                       'rows': records['row-token-audit.json'], 'no_GPU_or_quality': True,
                                       'training_token_ratio_vs_382000': summary['training_token_ratio_vs_382000']})
        print('NATIVE_FULLCONTEXT_CPU_AUDIT_COMPLETE_NO_GPU', flush=True)
    except BaseException as error:
        save('launcher-interruption.json', {'error_type': type(error).__name__, 'error_text': str(error), 'known_job': job,
                                            'intent_exists': (D / 'execution-intent.json').exists(),
                                            'action': 'preserve_known_job_or_ambiguous_intent_no_stop_no_relaunch'})
        raise


if __name__ == '__main__':
    main()
