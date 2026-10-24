"""One native CPU contract under an absolute 120+5 second process deadline."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path)
    with path.open('x') as handle:json.dump(value,handle,indent=2);handle.write('\n');handle.flush();os.fsync(handle.fileno())
    parent=path.resolve().parent
    while True:
        fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        if parent.parent==parent:break
        parent=parent.parent


def main():
    os.umask(0o077);plan=json.loads(Path('cpu-plan.json').read_text());start=time.monotonic();utc=time.time()
    assert plan['phase']=='cpu' and plan['cpu_authorized'] and plan['budget_seconds']==120 and plan['grace_seconds']==5
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    for path,item in plan['files'].items():assert Path(path).stat().st_size==item['bytes'] and sha(path)==item['sha256']
    assert not Path(plan['output_dir']).exists()
    assert not any(Path(n).exists() for n in ['driver-started.json','child-started.json','wrapper-complete.json','stdout.log','stderr.log'])
    save('driver-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),command=plan['argv'],plan_sha256=sha('cpu-plan.json'),deadline_TERM_unix=utc+120,deadline_KILL_unix=utc+125,budget_seconds=120,grace_seconds=5,no_automatic_retry=True))
    timeout=False
    with Path('stdout.log').open('xb') as out,Path('stderr.log').open('xb') as err:
        child=subprocess.Popen(plan['argv'],stdout=out,stderr=err,start_new_session=True)
        save('child-started.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),driver_pid=os.getpid(),child_pid=child.pid))
        try:rc=child.wait(timeout=max(.001,start+120-time.monotonic()))
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=max(.001,start+125-time.monotonic()))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
        for stream in [out,err]:stream.flush();os.fsync(stream.fileno())
    report=dict(at_utc=datetime.now(timezone.utc).isoformat(),returncode=rc,child_pid=child.pid,timed_out=timeout,
                seconds=time.monotonic()-start,budget_seconds=120,grace_seconds=5,no_automatic_retry=True)
    if rc==0 and not timeout:
        try:
            result=Path(plan['output_dir'])/'result.json';value=json.loads(result.read_text())
            assert value['status']=='PASS_NATIVE_FULL_INPUT_AND_INDEXED_POSITION_CPU' and value['train_rows']==87 and value['held_rows']==39
            assert value['all_local_native_actual_IDs_labels_rendering_exact'] and len(value['numerical'])==6 and value['no_weights_no_GPU_no_fit_no_quality']
            for key in ['runtime','tokenizer_assets','observations','encoded']:
                item=value[key];assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
            for item in value['numerical']:
                assert Path(item['path']).stat().st_size==item['bytes'] and sha(item['path'])==item['sha256']
                assert json.loads(Path(item['path']).read_text())['passes']
            report.update(all126_native_inputs_verified=True,result_sha256=sha(result),result_exists=True,no_GPU=True)
        except Exception as error:report['endpoint_error']=dict(type=type(error).__name__,message=str(error))
    save('wrapper-complete.json',report);print(json.dumps(report),flush=True)
    raise SystemExit(rc or (1 if timeout or 'endpoint_error' in report else 0))


if __name__=='__main__':main()
