"""Unique original901 four-stream owner, one shared1800+TERM30 clock."""
import argparse,json,os,shutil,signal,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from common import check_sources,ref,save,fsync_chain

def snapshot():
    gpu=subprocess.run(['nvidia-smi','--id=1','--query-gpu=index,uuid,name,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    return dict(at_utc=datetime.now(timezone.utc).isoformat(),gpu=gpu,compute_processes=processes)

def main():
    started=time.monotonic();ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args();pp=a.plan.resolve();d=pp.parent;p=json.loads(pp.read_text())
    assert p['execution_ready'] and (p['cap_seconds'],p['soft_stop_seconds'],p['term_grace_seconds'])==(1800,1770,30)
    assert p['flow_order']==['normal','reverse','public-new','patient'] and sum(p['flows'][n]['rows'] for n in p['flow_order'])==234
    assert not (d/'launch-started.json').exists() and not (d/'terminal.json').exists() and not (d/'actual').exists()
    check_sources(p,d);deadline=started+1800;soft=started+1770
    state=dict(status='RUNNING_UNIQUE_ORIGINAL901_ALL_FLA',started_utc=datetime.now(timezone.utc).isoformat(),source=ref(__file__),plan=ref(pp),cap_seconds=1800,soft_stop_seconds=1770,term_grace_seconds=30,absolute_monotonic_deadline=deadline,absolute_soft_deadline=soft,timeout_triggered=False,flow_order=p['flow_order'],flows={},expected_rows=234,cached_rows=0,gold_read=False,quality_calls=0)
    save(d/'launch-started.json',state);fsync_chain(d,Path(p['project_root']));child=None;success=False
    try:
        gate=snapshot();save(d/'fresh-GPU1.json',gate);f=[s.strip() for s in gate['gpu'].split(',')]
        assert len(f)==6 and f[:3]==['1',p['gpu_uuid'],'NVIDIA A30'] and int(f[3])<=100 and int(f[4])>=23000 and int(f[5])==24576
        assert not any(x.split(',')[0].strip()==p['gpu_uuid'] for x in gate['compute_processes'].splitlines())
        assert shutil.disk_usage(d).free>=1024**3
        env=dict(os.environ);env.update(p['env']);env['CUDA_VISIBLE_DEVICES']=p['gpu_uuid']
        assert env.get('PYTORCH_ALLOC_CONF') is None and env.get('PYTORCH_CUDA_ALLOC_CONF') is None
        for name in p['flow_order']:
            assert time.monotonic()<soft,'Shared soft deadline reached before next flow; no reset'
            command=[sys.executable,'-u',str(d/'infer_flow.py'),'--plan',str(pp),'--flow',name,'--soft-deadline-monotonic',str(soft)]
            flow_started=time.monotonic()
            with (d/(name+'.stdout.log')).open('xb') as stdout,(d/(name+'.stderr.log')).open('xb') as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,env=env,cwd=d,start_new_session=True)
                save(d/(name+'.child.json'),dict(pid=child.pid,owned_group=os.getpgid(child.pid),command=command,started_utc=datetime.now(timezone.utc).isoformat(),same_absolute_monotonic_deadline=deadline))
                try:code=child.wait(timeout=max(0,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    state['timeout_triggered']=True;os.killpg(child.pid,signal.SIGTERM)
                    try:code=child.wait(timeout=max(0,deadline+30-time.monotonic()))
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
                stdout.flush();os.fsync(stdout.fileno());stderr.flush();os.fsync(stderr.fileno())
            item=dict(exit_code=code,wall_seconds=time.monotonic()-flow_started,finished_utc=datetime.now(timezone.utc).isoformat(),deadline_not_reset=True)
            sp=d/'actual'/name/'summary.json'
            if sp.exists():item['summary']=ref(sp)
            state['flows'][name]=item;save(d/(name+'.exit.json'),item)
            if code!=0 or state['timeout_triggered']:raise RuntimeError('Flow failed; preserve originals, no automatic retry: '+name)
            summary=json.loads(sp.read_text());assert summary['status']=='COMPLETE_FRESH_ORIGINAL901_FLOW_NO_QUALITY' and summary['completed_rows']==p['flows'][name]['rows'] and summary['cached_rows']==0 and summary['parsed_rows']==0
        success=True;state['status']='COMPLETE_ORIGINAL901_FOUR_FRESH_FLA_FLOWS_NO_QUALITY';state['exit_code']=0
    except BaseException as e:
        state.update(status='FAILED_ORIGINAL901_ALL_FLA_PRESERVED_NO_RETRY',error_type=type(e).__name__,error=str(e),exit_code=1);raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=max(0,min(30,deadline+30-time.monotonic())))
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        try:
            state['release']=snapshot();state['target_UUID_has_no_compute_process']=not any(x.split(',')[0].strip()==p['gpu_uuid'] for x in state['release']['compute_processes'].splitlines())
        except BaseException as e:state['release_query_error_type']=type(e).__name__
        state['files']={str(q.relative_to(d)):{k:v for k,v in ref(q).items() if k!='path'} for q in sorted(d.rglob('*')) if q.is_file() and q.name!='terminal.json'}
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started);save(d/'terminal.json',state);fsync_chain(d,Path(p['project_root']));print(json.dumps({k:state.get(k) for k in ['status','exit_code','wall_seconds','timeout_triggered','target_UUID_has_no_compute_process']}),flush=True)
    return 0 if success else 1
if __name__=='__main__':raise SystemExit(main())
