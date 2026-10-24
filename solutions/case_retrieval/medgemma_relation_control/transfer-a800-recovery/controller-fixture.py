#!/usr/bin/env python3
"""Run byte-identical recovery controller with a fully local fake Run module."""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import tempfile
import time
import traceback
import types

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'run_once.py'
PRIOR = Path('/workspace/patientphex-runs/round2-20261023/research/candidate-order-warmstart-entry/run_paired_ack_paths.py')
OUTPUT = ROOT / 'controller-fixture.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def execute_case(name, source_bytes, controller_sha, prior_sha):
    events = []
    instances = []
    terminal = name == 'terminal_then_ack_failure'
    existing = name == 'existing_execution_state'
    with tempfile.TemporaryDirectory(prefix='medgemma-controller-fixture-') as temporary:
        root = Path(temporary)
        copied = root / 'run_once.py'
        copied.write_bytes(source_bytes)
        formal = root / 'formal'
        formal.mkdir()
        marker = formal / 'execution-state.json'
        if existing:
            marker.write_text('{"status":"already_started_synthetic"}\n')
        original_marker = marker.read_bytes() if existing else None
        check(sha(copied) == controller_sha, 'Controller copy must be byte-identical')
        plan = dict(controller_sha256=controller_sha, prior_controller_source_sha256=prior_sha,
                    local_evidence=[], copy_budget_seconds=300, source_mount_id='synthetic-source-mount',
                    target_mount_id='synthetic-target-mount', source_server='synthetic_server.py',
                    source_model='synthetic/model', source_expected='synthetic/expected.json',
                    remote_recovery='synthetic/recovery', resume_state_sha256='1' * 64,
                    original_manifest_sha256='2' * 64, target_fetcher='synthetic_fetch.py',
                    source_private_IP='192.0.2.1', destination='synthetic/destination',
                    preparation='synthetic/preparation', total_assets_bytes=19)
        ready = dict(ready=True, port=12345, manifest_sha256=plan['original_manifest_sha256'],
                     manifest={'files': {f'synthetic-{i}': {'bytes': 1, 'sha256': '3' * 64} for i in range(19)}})

        class FakeRun:
            def __init__(self, directory):
                check(directory == formal, 'Controller escaped temporary formal directory')
                self.plan = plan
                self.deadline = None
                instances.append(self)

            def save(self, path, value):
                path = Path(path)
                check(path.parent == formal, 'Unexpected fixture write path')
                with path.open('x') as handle:
                    handle.write(json.dumps(value, indent=2) + '\n')
                events.append({'operation': 'save', 'file': path.name})

            def start_bounded(self, mount, argv, cwd, env, record):
                check(mount in (plan['source_mount_id'], plan['target_mount_id']), 'Unexpected mount')
                job = {'id': 'known-source-server' if mount == plan['source_mount_id'] else 'known-target-fetch'}
                events.append({'operation': 'start_bounded', 'mount': mount, 'job': job})
                self.save(record, job)
                return job

            def call(self, mount, operation, arguments):
                events.append({'operation': operation, 'mount': mount, 'arguments': arguments})
                check(mount == plan['source_mount_id'], 'Only known source server may be called')
                check(arguments['process_id'] == 'known-source-server', 'Unknown process referenced')
                if operation == 'read_process_output':
                    return {'running': True, 'content': json.dumps(ready) + '\n'}
                if operation == 'stop_process':
                    return {'running': False, 'stopped': 'known-source-server'}
                raise AssertionError('Unstubbed operation: ' + operation)

            def watch(self, mount, job, directory):
                check(mount == plan['target_mount_id'] and job == {'id': 'known-target-fetch'}, 'Wrong watched job')
                events.append({'operation': 'watch', 'job': job, 'terminal_return': terminal})
                if not terminal:
                    raise ConnectionError('synthetic connection loss while fetch state is unknown')
                return {'running': False, 'exit_code': 0}

            def command(self, mount, code, timeout):
                check(terminal and mount == plan['target_mount_id'], 'Unexpected ACK command')
                events.append({'operation': 'ACK_readback', 'remote_code_executed': False})
                raise OSError('synthetic ACK readback failure after confirmed terminal fetch')

        fake = types.ModuleType('run_paired_ack_paths')
        fake.Run = FakeRun
        fake.sha = sha
        old_module = sys.modules.get('run_paired_ack_paths')
        old_path = list(sys.path)
        sys.modules['run_paired_ack_paths'] = fake
        caught = None
        try:
            runpy.run_path(str(copied), run_name='__main__')
        except BaseException as error:
            caught = error
        finally:
            sys.path[:] = old_path
            if old_module is None:
                sys.modules.pop('run_paired_ack_paths', None)
            else:
                sys.modules['run_paired_ack_paths'] = old_module
        check(len(instances) == 1, 'Real controller must use exactly the injected fake Run')
        starts = [event for event in events if event['operation'] == 'start_bounded']
        stops = [event for event in events if event['operation'] == 'stop_process']
        if existing:
            check(type(caught) is AssertionError, 'Existing state must reject before launch')
            check(not starts and not stops and events == [], 'Existing state must perform no work')
            check(marker.read_bytes() == original_marker, 'Existing state must remain unchanged')
            check(not (formal / 'controller-interruption.json').exists(), 'Guard must run before transfer body')
            evidence = {'existing_marker_unchanged': True, 'launches': 0, 'stop_process_calls': 0}
        else:
            expected = OSError if terminal else ConnectionError
            check(type(caught) is expected, 'Controller did not propagate expected synthetic exception')
            check(len(starts) == 2, 'Exactly one server and one fetch launch expected')
            interruption = json.loads((formal / 'controller-interruption.json').read_text())
            check(interruption['fetch_terminal_confirmed'] is terminal, 'Wrong terminal/unknown distinction')
            check(interruption['known_server'] == {'id': 'known-source-server'}, 'Lost known server ID')
            check(interruption['known_fetch'] == {'id': 'known-target-fetch'}, 'Lost known fetch ID')
            check(interruption['action'] == 'preserve_remote_jobs_when_fetch_state_unknown_no_relaunch', 'Wrong recovery action')
            check(interruption['error_type'] == expected.__name__, 'Exception evidence differs')
            check(time.time() < instances[0].deadline, 'Fixture accidentally exercised deadline expiry')
            check(not (formal / 'complete.json').exists(), 'A failed ACK/monitor must not claim completion')
            if terminal:
                check(len(stops) == 1 and stops[0]['arguments'] == {'process_id': 'known-source-server'},
                      'Confirmed terminal fetch may clean only the known source server')
                check((formal / 'server-stopped.json').exists(), 'Known-server cleanup receipt missing')
            else:
                check(not stops and not (formal / 'server-stopped.json').exists(),
                      'Connection loss must not stop either known remote job')
                check(not any(event['operation'] == 'ACK_readback' for event in events),
                      'Unknown fetch must not proceed to ACK readback')
            evidence = {'launches': len(starts), 'stop_process_calls': len(stops),
                        'fetch_terminal_confirmed': terminal, 'known_server': interruption['known_server'],
                        'known_fetch': interruption['known_fetch'], 'complete_claimed': False}
        return {'case': name, 'status': 'PASS', 'propagated_exception': type(caught).__name__,
                'evidence': evidence, 'events': events}


def main():
    started = time.monotonic()
    check(not OUTPUT.exists(), 'Do not overwrite an existing fixture receipt')
    source_bytes = SOURCE.read_bytes()
    controller_sha, prior_sha = sha(SOURCE), sha(PRIOR)
    actual_plan = ROOT / 'formal/run-plan.json'
    plan_sha = sha(actual_plan) if actual_plan.exists() else None
    result = {'status': 'RUNNING', 'source': str(SOURCE), 'controller_sha256': controller_sha,
              'source_bytes': len(source_bytes), 'prior_controller_sha256': prior_sha,
              'harness_sha256': sha(__file__), 'actual_plan_sha256_before': plan_sha,
              'method': 'byte-identical temporary controller via runpy; sys.modules fake Run injection',
              'real_MCP_calls': 0, 'real_HTTP_calls': 0, 'GPU_calls': 0, 'model_loads': 0, 'cases': []}
    try:
        for name in ('watch_connection_error', 'terminal_then_ack_failure', 'existing_execution_state'):
            result['cases'].append(execute_case(name, source_bytes, controller_sha, prior_sha))
        check(sha(SOURCE) == controller_sha, 'Original controller changed during fixture')
        check((sha(actual_plan) if actual_plan.exists() else None) == plan_sha, 'Original plan changed during fixture')
        result.update(status='PASS', source_and_plan_unchanged=True, seconds=time.monotonic() - started)
    except BaseException as error:
        result.update(status='FAIL', error_type=type(error).__name__, error_text=str(error),
                      traceback=traceback.format_exc(), seconds=time.monotonic() - started)
        with OUTPUT.open('x') as handle:
            handle.write(json.dumps(result, indent=2) + '\n')
        raise
    with OUTPUT.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'cases': len(result['cases']),
                      'seconds': result['seconds'], 'receipt': str(OUTPUT),
                      'receipt_sha256': sha(OUTPUT), 'controller_sha256': controller_sha}, indent=2))


if __name__ == '__main__':
    main()
