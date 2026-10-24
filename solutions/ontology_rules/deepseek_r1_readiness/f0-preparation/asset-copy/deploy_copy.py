"""Deploy only owned copy utilities, verify both exact sources; no server/fetch/GPU start."""
from pathlib import Path
import json,sys,hashlib,shlex,os,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
F=Path(__file__).parent;p=json.loads((F/'formal/run-plan.json').read_text());rel=p['remote_tools'];root='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 with (F/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def un(r):return r.get('structuredContent') or json.loads(r['content'][0]['text'])
save('deployment-intent.json',{'at':time.time(),'plan_sha256':sha(F/'formal/run-plan.json'),'no_model_copy_or_GPU_start':True});c=Client();c.initialize();receipts={}
for label,mount,uid,names in [('source',p['source_mount_id'],1006,['serve_assets.py']),('target',p['target_mount_id'],1045,['fetch_assets.py','expected-files.json'])]:
 expected={n:sha(F/n) for n in names}
 for n in names:
  r=c.request('tools/call',{'name':'write_file','arguments':dict(mount_id=mount,path=rel+'/'+n,content=(F/n).read_text(),encoding='utf8',create_parents=True)});save(label+'-'+n+'.upload.json',r);assert not r.get('isError')
 code="from pathlib import Path;import hashlib,json,os,shutil;p=Path("+repr(root+rel)+");expected="+repr(expected)+";assert p.stat().st_uid==os.getuid()=="+str(uid)+";os.chmod(p,0o700);a={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in expected};assert a==expected;[compile((p/n).read_text(),n,'exec') for n in expected if n.endswith('.py')];assert not (p/'server-ready.json').exists();print(json.dumps({'status':'ACTUAL_COPY_UTILITY_SHA_COMPILE_PASS_NO_START','uid':os.getuid(),'sources':a,'SHM_free':shutil.disk_usage('/dev/shm').free}))"
 r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=mount,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30)});save(label+'-deployment-check.json',r);assert not r.get('isError') and un(r)['exit_code']==0;receipts[label]=json.loads(un(r)['stdout'])
save('deployment-complete.json',receipts);print(json.dumps(receipts))
