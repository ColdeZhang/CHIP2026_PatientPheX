"""One explicitly authorized full58 A800 GPU2 process; null budget/cache/ACK never launches."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,time,traceback
from full58_contract import validate_plan

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,obj):
    p=Path(p);t=p.with_suffix('.tmp')
    with t.open('w') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
    fd=os.open(str(p.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def query():
    gpu=subprocess.run(['nvidia-smi','--id=2','--query-gpu=index,uuid,name,memory.used,memory.free,memory.total','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True,timeout=10).stdout.strip()
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True,timeout=10).stdout.strip()
    return dict(at=now(),gpu=gpu,compute_apps=processes)
def validate(gate,uuid):
    values=[x.strip() for x in gate['gpu'].split(',')];assert len(values)==6 and values[0]=='2' and values[1]==uuid=='GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc' and 'A800' in values[2]
    assert int(values[3])<=100 and int(values[4])>=80000 and int(values[5])==81920
    assert not any(line.split(',')[0].strip()==uuid for line in gate['compute_apps'].splitlines())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=Path(a.plan).resolve();base=p.parent;plan=json.loads(p.read_text());statepath=base/'wrapper-result.json'
    assert not statepath.exists() and not (base/'wrapper-started.json').exists(),'Original probe marker exists: reconcile; never launch twice'
    assert os.getuid()==1045
    validate_plan(plan)
    assert all(sha(base/name)==digest for name,digest in plan['sources'].items())
    assert plan['gpu_physical_index']==2 and plan['gpu_uuid']=='GPU-f516a193-de9c-8552-7ac5-4725f24cc0fc'
    assert plan['python']=='/home/zhangyuheng/projects/CHIP2026_PatientPheX/ontology-round2/qwen-venv/bin/python'
    assert plan['GPU_environment']=={'PYTHONPATH':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/qwen35-preparation/a800-migration/runtime'}
    assert all(isinstance(plan[key],dict) and set(plan[key])=={'path','bytes','sha256'} for key in ['CPU_result','all58_audit','all58_encoded','prompts_gzip','target_copy_ACK']), 'CPU/ACK refs must be frozen before launch'
    state=dict(status='starting_owned_probe',at=now(),source_sha256=sha(__file__),plan_sha256=sha(p),cap_seconds=plan['process_budget_seconds'],TERM_grace_seconds=plan['TERM_grace_seconds'],fixed58_whole_response_contract=True,new_generation_target=58-len(plan['cached_indices']),reuse_target=len(plan['cached_indices']),no_gold_quality=True)
    atomic(base/'wrapper-started.json',state)
    proc=None
    try:
        gate=query();state['fresh_GPU2']=gate;validate(gate,plan['gpu_uuid']);atomic(base/'fresh-GPU2.json',gate)
        env=dict(os.environ);env.update(plan['GPU_environment']);env.update(CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
        command=['timeout','--signal=TERM','--kill-after=%ss'%plan['TERM_grace_seconds'],'%ss'%plan['process_budget_seconds'],plan['python'],'-u',str(base/'infer_full58.py'),'--plan',str(p)];state['command']=command;state['GPU_main_started_at']=now();t=time.monotonic()
        with (base/'probe.stdout.log').open('x') as out,(base/'probe.stderr.log').open('x') as err:
            proc=subprocess.Popen(command,cwd=base,env=env,stdout=out,stderr=err,start_new_session=True);state['timeout_pid']=proc.pid;state['owned_process_group']=os.getpgid(proc.pid);assert state['owned_process_group']==proc.pid;atomic(base/'wrapper-running.json',state);rc=proc.wait()
        state.update(returncode=rc,GPU_main_seconds=time.monotonic()-t,GPU_main_finished_at=now(),status='owned_probe_process_completed')
        terminal=base/'terminal/result.json';state['result_exists']=terminal.exists()
        if terminal.exists():state['result_sha256']=sha(terminal);state['result_status']=json.loads(terminal.read_text())['status']
    except BaseException as e:state.update(status='failed_wrapper_preserved',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        if proc is not None and proc.poll() is None:
            import signal
            try:
                assert state['owned_process_group']==proc.pid and os.getpgid(proc.pid)==proc.pid
                os.killpg(proc.pid,signal.SIGTERM)
                try:state['interrupted_owned_process_returncode']=proc.wait(timeout=plan['TERM_grace_seconds'])
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL);state['interrupted_owned_process_returncode']=proc.wait(timeout=5)
                state['only_owned_process_group_cleanup']=True
            except ProcessLookupError:
                state['owned_process_already_exited_during_cleanup']=True;state['interrupted_owned_process_returncode']=proc.poll()
        try:state['actual_release']=query();state['target_UUID_has_no_compute_process']=not any(line.split(',')[0].strip()==plan['gpu_uuid'] for line in state['actual_release']['compute_apps'].splitlines())
        except BaseException as e:state['release_query_error']=repr(e)
        state['finished_at']=now();atomic(statepath,state);print(json.dumps({k:state.get(k) for k in ['status','returncode','GPU_main_seconds','result_status','target_UUID_has_no_compute_process']}),flush=True)
if __name__=='__main__':main()
