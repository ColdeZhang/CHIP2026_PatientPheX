"""One bounded probe -> fresh formal -> full inference controller, no retries."""
import argparse,json,os,subprocess,time
from pathlib import Path
from helpers import write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--python',required=True);a=ap.parse_args();plan=json.loads(Path(a.plan).read_text());root=Path(plan['run_root']);jobs=root/'jobs';jobs.mkdir(parents=True,exist_ok=True)
    write(jobs/'controller-start.json',{'unix':time.time(),'pid':os.getpid(),'plan':a.plan,'budgets':plan['budgets']})
    for phase in ['probe','formal','infer']:
        job=jobs/phase;job.mkdir(exist_ok=False);start=time.monotonic();command=['timeout','--signal=TERM','--kill-after=30s',str(plan['budgets'][phase])+'s',a.python,'-u',str(Path(__file__).parent/'gpu_entry.py'),'--plan',a.plan,'--mode',phase];write(job/'start.json',{'at_unix':time.time(),'command':command,'budget_seconds':plan['budgets'][phase],'grace_seconds':30})
        with (job/'stdout.txt').open('x') as out,(job/'stderr.txt').open('x') as err:
            child=subprocess.Popen(command,stdout=out,stderr=err);write(job/'child.json',{'pid':child.pid});rc=child.wait()
        release={'at_unix':time.time(),'child_exit_code':rc,'seconds':time.monotonic()-start,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True),'apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)};write(job/'ended.json',release)
        if rc:write(jobs/'controller-complete.json',{'status':'FAILED_NO_RETRY','phase':phase,'child_exit_code':rc});raise SystemExit(rc)
        result=json.loads((root/phase/'complete.json').read_text())
        if phase=='probe' and not result['formal_budget_gate']:
            write(jobs/'controller-complete.json',{'status':'COMPLETE_PROBE_COST_GATE_CLOSED','cost':result['cost'],'new_quality':0});return
    write(jobs/'controller-complete.json',{'status':'COMPLETE_ALL_THREE_GPU_PHASES','at_unix':time.time(),'no_quality_or_gold':True})

if __name__=='__main__':main()
