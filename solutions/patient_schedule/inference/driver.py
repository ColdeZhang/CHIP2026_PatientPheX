"""One fresh80 inference after four durable epochs, no automatic retry."""
from pathlib import Path
import argparse,json,os,subprocess,time
from epoch_driver import sha,save,gpu

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());root=Path(plan['root']);control=root/'inference-control';control.mkdir(exist_ok=False)
 assert plan['cap_seconds']==1200 and plan['grace_seconds']==30
 for n,r in plan['files'].items():assert Path(n).stat().st_size==r['bytes'] and sha(n)==r['sha256']
 assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
 fresh=gpu(plan);ix,uuid,used,free,total=[x.strip() for x in fresh['target_csv'].split(',')];save(control/'fresh-gpu.json',fresh)
 assert uuid==plan['gpu_uuid'] and int(used)<=100 and int(free)>=78000 and not fresh['target_compute_rows']
 assert not Path(plan['output']).exists() and os.statvfs(root).f_bavail*os.statvfs(root).f_frsize>=64*1024**2
 started=time.time();save(control/'driver-started.json',dict(started_unix=started,deadline=started+1200,plan_sha256=sha(a.plan),pid=os.getpid()))
 with (control/'stdout.log').open('xb') as out,(control/'stderr.log').open('xb') as err:
  child=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=30s','1200s']+plan['argv'],stdout=out,stderr=err)
  save(control/'child-started.json',dict(pid=child.pid));rc=child.wait()
 report=dict(returncode=rc,seconds=time.time()-started,deadline=started+1200)
 if rc==0:
  try:
   output=Path(plan['output']);m=json.loads(output.with_suffix('.manifest.json').read_text());rows=[json.loads(x) for x in output.read_text().splitlines()]
   assert len(rows)==len({x['row_id'] for x in rows})==80 and m['strict_all80_EOS_and_PID_CID_schema'] and m['fallback_rows']==0 and m['output_sha256']==sha(output)
   report['complete80']=True;report['responses_sha256']=sha(output)
  except Exception as e:report['endpoint_error']=repr(e)
 save(control/'wrapper-complete.json',report);save(control/'gpu-release.json',gpu(plan));raise SystemExit(rc or int('endpoint_error' in report))
if __name__=='__main__':main()
