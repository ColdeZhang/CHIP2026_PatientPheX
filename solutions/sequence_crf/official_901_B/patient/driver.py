"""Run one frozen B patient CPU preparation or inference with an actual deadline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,v):
    p=Path(p)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(str(p.parent),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args()
    plan=json.loads(Path(a.plan).read_text());phase=plan['phase'];assert phase in ('prepare','infer')
    cap=plan['cap_seconds'];grace=plan['grace_seconds'];assert isinstance(cap,int) and 0<cap<=7200 and grace==30
    control=Path(plan['control']);control.mkdir(parents=True,exist_ok=False)
    current=control
    while True:
        fd=os.open(str(current),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        if current==Path(plan['project']):break
        assert current!=current.parent, 'control must be below own persistent project'
        current=current.parent
    start=time.time();save(control/'driver-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),plan_sha256=sha(a.plan),TERM_unix=start+cap,KILL_unix=start+cap+grace,once_only=True))
    for name,item in plan['files'].items():assert Path(name).stat().st_size==item['bytes'] and sha(name)==item['sha256'],name
    output=Path(plan['output']);assert not output.exists()
    if phase=='infer':
        assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpu_uuid']
        mem=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        target=[s for s in mem.splitlines() if s.split(',')[0].strip()==plan['gpu_uuid']]
        assert len(target)==1 and int(target[0].split(',')[1])<=64 and not any(s.split(',')[0].strip()==plan['gpu_uuid'] for s in apps.splitlines())
        save(control/'fresh-gpu.json',dict(memory=target,apps=apps,empty=True))
    remaining=cap-(time.time()-start);assert remaining>0
    with (control/'stdout.log').open('xb') as out,(control/'stderr.log').open('xb') as err:
        child=subprocess.Popen(plan['argv'],stdout=out,stderr=err,start_new_session=True)
        save(control/'child-started.json',dict(pid=child.pid,process_group=child.pid,TERM_unix=start+cap,KILL_unix=start+cap+grace))
        timed_out=False;killed=False
        try:rc=child.wait(timeout=max(0., start+cap-time.time()))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=grace)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait();killed=True
    report=dict(at_utc=datetime.now(timezone.utc).isoformat(),phase=phase,returncode=rc,seconds=time.time()-start,timed_out=timed_out,killed=killed,no_retry=True,no_fit=True,no_score=True)
    endpoint_error=None
    if rc==0 and not timed_out:
        try:
            manifest=output/('manifest.json' if phase=='prepare' else 'responses.manifest.json');m=json.loads(manifest.read_text());report['manifest_sha256']=sha(manifest)
            if phase=='prepare':assert m['native_all_ids_exact'] and m['native_all_decode_exact'] and m['transformation_inverse_all_exact'];report['rows']=m['rows']
            else:
                rows=[json.loads(x) for x in (output/'responses.jsonl').read_text().splitlines()]
                assert len(rows)==len({r['row_id'] for r in rows})==plan['rows']
                assert m['all_B_raw_saved_before_codec'] and m['all_rows_fresh'] and m['no_cache_or_resume'] and m['no_baseline_fallback']
                assert m['cached_rows']==0 and m['new_rows']==plan['rows'] and sum(m['codec_status_counts'].values())==plan['rows']
                for k in ['decisions','raw_index']:assert sha(m[k]['path'])==m[k]['sha256']
                assert sha(output/'responses.jsonl')==m['output_sha256']
                report.update(rows=len(rows),responses_sha256=m['output_sha256'],codec_status_counts=m['codec_status_counts'],complete_coverage=True)
                save(output/'complete.json',report)
        except Exception as e:endpoint_error=dict(type=type(e).__name__,message=str(e));report['endpoint_error']=endpoint_error
    if phase=='infer':
        mem=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        report['gpu_release']=dict(at_utc=datetime.now(timezone.utc).isoformat(),memory=[s for s in mem.splitlines() if s.split(',')[0].strip()==plan['gpu_uuid']],no_process=not any(s.split(',')[0].strip()==plan['gpu_uuid'] for s in apps.splitlines()))
    save(control/'wrapper-complete.json',report);print(json.dumps(report),flush=True)
    raise SystemExit(rc or (1 if endpoint_error or timed_out else 0))

if __name__=='__main__':main()
