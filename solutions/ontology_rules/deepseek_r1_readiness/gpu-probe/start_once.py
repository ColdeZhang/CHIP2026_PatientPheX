"""Unique conditional start; explicit intent survives UNKNOWN outcome."""
from pathlib import Path
import json,sys,hashlib,shlex,time,os
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
G=Path(__file__).parent;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/deepseek-r1-0528-qwen3-8b-readiness/gpu-probe-v1';root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,obj):
 with (G/n).open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def main():
 assert not (G/'start-intent.json').exists() and not (G/'plan-upload-intent.json').exists(),'Reconcile previous write/start before any retry'
 plan=json.loads((G/'plan.json').read_text());assert plan['execution_ready'] is True
 save('plan-upload-intent.json',dict(at=time.time(),plan_sha256=sha(G/'plan.json'),on_unknown='Inventory actual plan bytes; never overwrite different or repeat start'))
 c=Client();c.initialize();r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=M,path=rel+'/plan.json',content=(G/'plan.json').read_text(),encoding='utf8',create_parents=True)});save('plan-upload-receipt.json',r);assert not r.get('isError')
 code="from pathlib import Path;import json,hashlib,os;p=Path("+repr(root+rel)+");s=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();assert s(p/'plan.json')=="+repr(sha(G/'plan.json'))+";d=json.loads((p/'plan.json').read_text());assert all(s(p/n)==h for n,h in d['sources'].items());assert all(Path(d[k]['path']).stat().st_size==d[k]['bytes'] and s(Path(d[k]['path']))==d[k]['sha256'] for k in ['assets_result','header_inventory','expected_files','CPU_result','yarn_result','selected_prompts','selected_encoded','derived_shapes']);assert not (p/'wrapper-started.json').exists() and not (p/'terminal').exists();print(json.dumps({'status':'ACTUAL_SOURCE_PLAN_INPUT_ASSET_TERMINAL_REFS_MATCH_NO_START','uid':os.getuid(),'plan_sha256':s(p/'plan.json')}))"
 r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),timeout_seconds=30)});save('actual-prestart-check.json',r);assert not r.get('isError') and un(r)['exit_code']==0
 args=dict(mount_id=M,cwd=rel,command=shlex.quote(plan['python'])+' -u launch_probe.py --plan plan.json',env={**plan['GPU_environment'],'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 save('start-intent.json',dict(at=time.time(),arguments=args,plan_sha256=sha(G/'plan.json'),original_unique_probe=True,on_unknown='Read remote wrapper-started/running/result and original managed job; never launch twice'))
 try:
  r=c.request('tools/call',{'name':'start_process','arguments':args});save('job.json',r);print(json.dumps(un(r)))
 except Exception as e:save('start-unknown-outcome.json',dict(at=time.time(),error_type=type(e).__name__,original_remote_process_may_be_running=True));raise
if __name__=='__main__':main()
