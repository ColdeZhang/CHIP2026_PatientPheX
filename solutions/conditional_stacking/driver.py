"""Exclusive one-fit local controller with native timeout and durable stage receipts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent/'reuse'))
import common as c


def run(activation):
    a=json.loads(Path(activation).read_text());p=c.read_json(a['prepared_plan'])
    assert a['execution_authorized'] and a['fit_calls_maximum']==1
    for r in a['sources'].values():c.checked(r)
    root=c.mkdir_new(p['output'])
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    c.write_json(root/'driver-started.json',dict(at_utc=c.now(),activation=c.ref(activation),pid=os.getpid(),
        command=sys.argv,python=sys.executable,fit_seconds=60,fit_grace=5))
    for mode in ['fit','assemble','quality']:
        cap=60 if mode=='fit' else 120
        cmd=['timeout','--signal=TERM','--kill-after=5',str(cap),a['runtime']['python'],str(Path(__file__).with_name('runner.py')),'--activation',str(Path(activation).resolve()),mode]
        c.write_json(root/(mode+'-intent.json'),dict(at_utc=c.now(),command=cmd,cap_seconds=cap,TERM_grace=5))
        start=time.monotonic()
        with (root/(mode+'.stdout')).open('xb') as stdout,(root/(mode+'.stderr')).open('xb') as stderr:
            result=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=environment)
            stdout.flush();os.fsync(stdout.fileno());stderr.flush();os.fsync(stderr.fileno())
        rc=c.write_json(root/(mode+'-process.json'),dict(at_utc=c.now(),return_code=result.returncode,
            seconds=time.monotonic()-start,timed_out=result.returncode in [124,137],stdout=c.ref(root/(mode+'.stdout')),stderr=c.ref(root/(mode+'.stderr'))))
        if result.returncode:
            c.write_json(root/'driver-failed.json',dict(at_utc=c.now(),phase=mode,process=rc,no_automatic_retry=True))
            raise SystemExit(result.returncode)
    c.write_json(root/'complete.json',dict(status='COMPLETE_FIXED_CPU_CONTROL',at_utc=c.now(),activation=c.ref(activation),
        fit=c.ref(root/'fit/complete.json'),F0_freeze=c.ref(root/'F0/freeze.json'),quality=c.ref(root/'quality/result.json'),
        fit_calls=1,GPU_calls=0))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--activation',required=True);args=ap.parse_args();run(args.activation)
