"""One bounded CPU stage; save the actual subprocess return code and terminal time."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time

R = Path(__file__).resolve().parent
stage = sys.argv[1]
settings = {
    'shape': ('inspect_complete.py', 'shape-plan.json', 120),
    'repair': ('repair_fasttext_header.py', 'repair-plan.json', 120),
    'native': ('check_native.py', 'native-cpu-plan.json', 180),
}
source, plan, cap = settings[stage]
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
output = R / (stage + '-job-state.json')
assert not output.exists(), 'Do not repeat a completed or running stage'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
job = {'stage': stage, 'started_at': now(), 'cap_seconds': cap,
       'TERM_grace_seconds': 15, 'source_sha256': sha(R/source),
       'plan_sha256': sha(R/plan), 'wrapper_sha256': sha(Path(__file__)),
       'command': ['timeout', '--signal=TERM', '--kill-after=15s', str(cap)+'s', sys.executable, source]}
output.write_text(json.dumps(job, indent=2)+'\n')
start = time.monotonic()
with (R/(stage+'.stdout.log')).open('xb') as stdout, (R/(stage+'.stderr.log')).open('xb') as stderr:
    rc = subprocess.run(job['command'], cwd=R, stdout=stdout, stderr=stderr).returncode
job.update(returncode=rc, elapsed_seconds=time.monotonic()-start, finished_at=now())
output.write_text(json.dumps(job, indent=2)+'\n')
raise SystemExit(rc)
