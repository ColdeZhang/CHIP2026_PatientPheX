"""Deploy an already checkpointed, SHA-bound attempt2; do not alter attempt1."""
from pathlib import Path
import base64
import hashlib
import json
import shlex
import sys

R = Path(__file__).resolve().parent
sys.path.insert(0, '/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
plan = json.loads((R / 'resume-plan.json').read_text())
assert plan['status'] == 'authorized_frozen_resume_after_attempt1_exit'
assert plan['budget_seconds'] == 1800 and plan['prior_job_terminal']
assert not (R / 'managed-job.json').exists()
client = Client()
client.initialize()
mount = plan['mount_id']
remote = plan['remote_cwd']


def call(name, values):
    result = client.request('tools/call', {'name': name, 'arguments': {'mount_id': mount, **values}})
    assert not result.get('isError'), result
    return result['structuredContent']


env = {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '', 'PYTHONNOUSERSITE': '1',
       'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
check = 'from pathlib import Path;assert not Path(' + repr(remote) + ').exists();print("fresh_owned_resume_directory")'
result = call('run_command', {'command': 'public-phenotagger-reuse/.venv/bin/python -c ' + shlex.quote(check),
                             'cwd': '.', 'env': env, 'timeout_seconds': 10})
assert result['exit_code'] == 0, result
files = ['resume_assets.py', 'launch_resume.py', 'resume-plan.json', 'attempt1-terminal.json']
digests = {name: hashlib.sha256((R / name).read_bytes()).hexdigest() for name in files}
for name in files:
    call('write_file', {'path': remote + '/' + name, 'content': base64.b64encode((R / name).read_bytes()).decode(),
                        'encoding': 'base64', 'create_parents': True})
check = 'from pathlib import Path;import hashlib,json;expected=' + repr(digests) + ';actual={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert actual==expected;print(json.dumps(actual))'
result = call('run_command', {'command': '../../public-phenotagger-reuse/.venv/bin/python -c ' + shlex.quote(check),
                             'cwd': remote, 'env': env, 'timeout_seconds': 10})
assert result['exit_code'] == 0, result
(R / 'deployment.json').write_text(json.dumps({'source_sha256': digests, 'remote_actual': json.loads(result['stdout'])}, indent=2) + '\n')
command = '../../public-phenotagger-reuse/.venv/bin/python launch_resume.py'
job = call('start_process', {'command': command, 'cwd': remote, 'env': env})
(R / 'managed-job.json').write_text(json.dumps({'command': command, 'env': env, 'sources': digests,
    'start_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'result': job}, indent=2) + '\n')
print(json.dumps(job), flush=True)
