"""Reserve and execute one CPU-only preparation/fit/finalize chain; no retries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    assert plan['execution_ready'] and plan['controller'] == ref(__file__)
    directory = Path(plan['output_dir'])
    state = dict(status='running',start_utc=now(),controller=ref(__file__),plan=ref(args.plan),
                 pid=os.getpid(),stages=[],no_retries=True)
    state_path = directory/'execution-state.json'
    write(state_path,state,exclusive=True)
    environment = os.environ.copy();environment.update(plan['environment'])
    started = time.monotonic()
    failure = None
    try:
        for stage,cap in [('prepare',60),('fit',60),('finalize',30)]:
            command = [plan['python'],plan['source']['path'],'--plan',str(Path(args.plan).resolve()),'--stage',stage]
            entry = dict(stage=stage,command=command,start_utc=now(),cap_seconds=cap)
            state['stages'].append(entry);write(state_path,state)
            tick = time.monotonic()
            with (directory/(stage+'.stdout.log')).open('xb') as stdout, (directory/(stage+'.stderr.log')).open('xb') as stderr:
                result = subprocess.run(command,env=environment,stdout=stdout,stderr=stderr,timeout=cap)
            entry.update(returncode=result.returncode,elapsed_seconds=time.monotonic()-tick,finish_utc=now(),
                         stdout=ref(directory/(stage+'.stdout.log')),stderr=ref(directory/(stage+'.stderr.log')))
            write(state_path,state)
            if result.returncode:
                raise RuntimeError(stage+' failed; no automatic rerun')
        state['status'] = 'complete'
    except BaseException as error:
        failure = error
        state.update(status='failed_preserve_existing_state_no_retry',error=repr(error))
    state.update(finish_utc=now(),elapsed_seconds=time.monotonic()-started)
    write(state_path,state)
    write(directory/'process-terminal.json',state,exclusive=True)
    print(json.dumps(dict(status=state['status'],elapsed_seconds=state['elapsed_seconds'])))
    if failure:
        raise failure


if __name__ == '__main__':
    main()
