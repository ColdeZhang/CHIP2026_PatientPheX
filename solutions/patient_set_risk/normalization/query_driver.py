"""Exclusive source-bound query admission and shared absolute deadline."""
import argparse,signal,subprocess,time
from pathlib import Path
from common import *

def resources():
    def q(x):return subprocess.check_output(['nvidia-smi',x,'--format=csv,noheader,nounits'],text=True,timeout=10)
    s=os.statvfs('.')
    return dict(at_unix=time.time(),gpu=q('--query-gpu=index,uuid,memory.used,memory.free,memory.total'),
                processes=q('--query-compute-apps=gpu_uuid,pid,used_memory'),disk_free_bytes=s.f_bavail*s.f_frsize)

def main():
    os.umask(0o077);ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--plan-sha',required=True);a=ap.parse_args()
    assert ref(a.plan)['sha256']==a.plan_sha;p=load(a.plan);root=Path(p['remote_root']);assert Path.cwd().resolve()==root.resolve()
    assert p['schema']=='scst-train-only-public-queries-v1' and p['authorized'] and p['budget_seconds']==180 and p['grace_seconds']==30
    assert p['shared_KILL_unix']==p['shared_TERM_unix']+30 and p['query_count']>0
    def admission():
        now=time.time();assert p['issued_unix']-5<=now<=p['admission_expires_unix']<=p['issued_unix']+300
        assert now+p['cost_estimate_seconds']+5<p['shared_TERM_unix'],'Insufficient original shared window'
    admission()
    for name,item in p['files'].items():verify(item,name)
    assert os.environ['CUDA_VISIBLE_DEVICES']==p['gpu_uuid'] and not Path(p['output_dir']).exists()
    gate=resources();save('fresh-resource-gate.json',gate)
    target=[s.split(',') for s in gate['gpu'].splitlines() if p['gpu_uuid'] in s]
    assert len(target)==1 and int(target[0][2])<=10 and int(target[0][3])>=20000 and p['gpu_uuid'] not in gate['processes'] and gate['disk_free_bytes']>=2*1024**3
    admission();started=time.time();mono=time.monotonic();term=min(started+180,p['shared_TERM_unix']);kill=min(term+30,p['shared_KILL_unix'])
    # Exclusive immutable publication is the fence: never restart/overwrite an UNKNOWN launch.
    save('driver-started.json',dict(plan=ref(a.plan),source=ref(__file__),driver_pid=os.getpid(),started_unix=started,
         deadline_TERM_unix=term,deadline_KILL_unix=kill,shared_TERM_unix=p['shared_TERM_unix'],shared_KILL_unix=p['shared_KILL_unix'],
         no_automatic_retry=True,resource_PASS=True))
    admission();timed_out=False
    cmd=[p['python'],'-u',p['query_source'],'--plan',str(Path(a.plan).resolve())]
    with Path('stdout.log').open('xb') as stdout,Path('stderr.log').open('xb') as stderr:
        child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,start_new_session=True)
        save('child-started.json',dict(pid=child.pid,driver_pid=os.getpid(),argv=cmd,pgid=os.getpgid(child.pid),start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]))
        try:rc=child.wait(timeout=max(.001,min(term-time.time(),mono+term-started-time.monotonic())))
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=max(.001,min(kill-time.time(),mono+kill-started-time.monotonic())))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
        for f in (stdout,stderr):f.flush();os.fsync(f.fileno())
    report=dict(returncode=rc,timed_out=timed_out,child_pid=child.pid,child_absent=not Path(f'/proc/{child.pid}').exists(),seconds=time.monotonic()-mono,plan=ref(a.plan))
    if rc==0 and not timed_out:
        try:
            out=Path(p['output_dir']);c=load(out/'complete.json')
            assert c['status']=='COMPLETE_SCST_PUBLIC_TRAIN_QUERIES' and c['query_count']==p['query_count'] and c['public_bank_rows']==41099
            assert c['plan']['sha256']==a.plan_sha and c['source']['sha256']==p['files'][p['query_source']]['sha256']
            for k in ('scores','vectors','counts','encoder'):verify(c[k]);assert Path(c[k]['path']).parent==out
            report.update(complete=ref(out/'complete.json'),complete_necessary_query_members_verified=True)
        except Exception as e:report['endpoint_error']=dict(type=type(e).__name__,message=str(e))
    report['release']=resources();report['target_has_no_process']=p['gpu_uuid'] not in report['release']['processes']
    report['success']=rc==0 and not timed_out and 'endpoint_error' not in report and report['child_absent'] and report['target_has_no_process']
    save('wrapper-complete.json',report);print(json.dumps(report));raise SystemExit(0 if report['success'] else 1)

if __name__=='__main__':main()
