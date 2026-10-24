from pathlib import Path
import sys,json,shlex,concurrent.futures
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
F=Path(__file__).parent
jobs={'source':('5a9955da-5240-41cc-9e52-05d3d8da8a3c',1006,'10.5.2.100'),'target':('69b991ee-183f-4add-af1a-579af92479c9',1045,'10.5.4.7')}
def one(label,item):
 mount,uid,ip=item
 code="""from pathlib import Path
import os,json,subprocess,shutil,datetime,hashlib
project=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX');assert os.getuid()==__UID__ and project.stat().st_uid==__UID__
ips=subprocess.check_output(['hostname','-I'],text=True).split();assert __IP__ in ips
own=Path('/dev/shm/patientphex-r1-f0-1045-20261024');matches=[str(p) for p in Path('/dev/shm').glob('patientphex*r1*')]
out={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'uid':os.getuid(),'project_uid':project.stat().st_uid,'private_IP':__IP__,'project_free':shutil.disk_usage(project).free,'SHM_free':shutil.disk_usage('/dev/shm').free,'target_namespace_exists':own.exists(),'matching_owned_R1_names':matches,'no_model_GPU_or_writes':True}
if __LABEL__=='source':
 base=project/'research/deepseek-r1-0528-qwen3-8b-readiness';expected=base/'assets/expected-files.json';assert hashlib.sha256(expected.read_bytes()).hexdigest()=='ee707db3a5b314c94c213386c2c8b8640fd9406759e68f9e729640f70ccbfaec';e=json.loads(expected.read_text());model=base/'models/DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6';out['source_model']=str(model);out['source_assets']={r['name']:{'exists':(model/r['name']).is_file(),'bytes':(model/r['name']).stat().st_size,'uid':(model/r['name']).stat().st_uid,'mtime_ns':(model/r['name']).stat().st_mtime_ns,'inode':(model/r['name']).stat().st_ino} for r in e['files']};assert all(out['source_assets'][r['name']]['bytes']==r['bytes'] for r in e['files'])
else:
 assert out['SHM_free']>20*1024**3
 if own.exists():out['existing_own_files']=[{'path':str(p.relative_to(own)),'bytes':p.stat().st_size} for p in own.rglob('*') if p.is_file()]
print(json.dumps(out))
""".replace('__UID__',str(uid)).replace('__IP__',repr(ip)).replace('__LABEL__',repr(label))
 compile(code,'remote_preflight','exec')
 c=Client();c.initialize();r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=mount,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30)})
 with (F/(label+'-preflight-tool.json')).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 assert not r.get('isError');v=r.get('structuredContent') or json.loads(r['content'][0]['text']);assert v['exit_code']==0,v
 out=json.loads(v['stdout']);(F/(label+'-preflight.json')).write_text(json.dumps(out,indent=2)+'\n');return label,out
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
 for label,out in ex.map(lambda x:one(*x),jobs.items()):print(json.dumps({'server':label,**out}),flush=True)
