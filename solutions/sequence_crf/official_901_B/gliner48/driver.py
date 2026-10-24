"""One fixed GLiNER48 B100 inference, bounded 300 seconds plus TERM30."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    args = parser.parse_args(); plan = json.loads(Path(args.plan).read_text()); started = time.time()
    assert plan['cap_seconds'] == 300 and plan['grace_seconds'] == 30
    control = Path(plan['control']); control.mkdir(exist_ok=False)
    save(control/'driver-started.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
        pid=os.getpid(), plan_sha256=digest(args.plan), TERM_unix=started+300, KILL_unix=started+330))
    for path, item in plan['files'].items():
        if 'bytes' in item: assert Path(path).stat().st_size == item['bytes']
        assert digest(path) == item['sha256'], path
    docs = [json.loads(x) for x in Path(plan['input']).read_text().splitlines()]
    ids = [str(x['pmc_id']) for x in docs]
    assert len(ids) == len(set(ids)) == 100 and sum(len(x['patient']) for x in docs) == 244
    assert all('entities' not in x and 'association' not in x for x in docs)
    assert not Path(plan['output']).exists()
    assert shutil.disk_usage(plan['project']).free >= 2*1024**3
    gpus = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
    target = [x for x in gpus.splitlines() if x.split(',')[0].strip() == plan['gpu_uuid']]
    assert len(target) == 1 and int(target[0].split(',')[1].strip()) <= 64
    assert not any(x.split(',')[0].strip() == plan['gpu_uuid'] for x in processes.splitlines())
    save(control/'fresh-gate.json', dict(gpus=gpus, processes=processes, target=plan['gpu_uuid'], passed=True))
    remaining = int(300-(time.time()-started)); assert remaining > 0
    with (control/'stdout.log').open('xb') as out, (control/'stderr.log').open('xb') as err:
        result = subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=30s',str(remaining)+'s']+plan['argv'], stdout=out, stderr=err)
    terminal = dict(at_utc=datetime.now(timezone.utc).isoformat(), returncode=result.returncode,
        seconds=time.time()-started, no_retry=True, no_fit=True, no_score=True)
    if result.returncode == 0:
        root = Path(plan['output']); manifest = json.loads((root/'manifest.json').read_text())
        assert manifest['source_sha256'] == plan['runner_sha256']
        assert manifest['input_sha256'] == plan['input_sha256']
        assert manifest['stats']['documents'] == 100 and manifest['strict_state_dict']
        assert manifest['all_source_word_indices_covered'] and manifest['all_actual_word_masks_and_token_counts_match_no_truncation']
        rows = [json.loads(x) for x in (root/'candidates.jsonl').read_text().splitlines()]
        assert [str(x['pmc_id']) for x in rows] == ids
        assert manifest['candidates_sha256'] == digest(root/'candidates.jsonl')
        terminal['manifest_sha256'] = digest(root/'manifest.json')
        terminal['candidate_sha256'] = manifest['candidates_sha256']
        terminal['documents'] = len(rows)
    gpus = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
    terminal['gpu_release'] = dict(at_utc=datetime.now(timezone.utc).isoformat(), target=plan['gpu_uuid'],
        target_memory=[x for x in gpus.splitlines() if x.split(',')[0].strip() == plan['gpu_uuid']],
        target_has_no_process=not any(x.split(',')[0].strip() == plan['gpu_uuid'] for x in processes.splitlines()))
    save(control/'wrapper-complete.json', terminal); print(json.dumps(terminal))


if __name__ == '__main__': main()
