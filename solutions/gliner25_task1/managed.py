"""Run one bounded child and persist its actual exit plus post-child GPU state."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from probe import write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);ap.add_argument('--seconds',type=int,required=True);ap.add_argument('--grace',type=int,default=30);ap.add_argument('command',nargs=argparse.REMAINDER);args=ap.parse_args()
    root=Path(args.directory);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'started.json').exists() and args.command
    command=args.command[1:] if args.command[0]=='--' else args.command
    started=time.monotonic();write(root/'started.json',{'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'budget_seconds':args.seconds,'grace_seconds':args.grace})
    with (root/'stdout.txt').open('w') as out,(root/'stderr.txt').open('w') as err:
        p=subprocess.run(['/usr/bin/timeout','--signal=TERM',f'--kill-after={args.grace}s',f'{args.seconds}s']+command,stdout=out,stderr=err)
    report={'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'child_exit_code':p.returncode,'elapsed_seconds':time.monotonic()-started,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True),'apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)}
    write(root/'ended.json',report);print(json.dumps(report),flush=True);sys.exit(p.returncode)


if __name__=='__main__':main()
