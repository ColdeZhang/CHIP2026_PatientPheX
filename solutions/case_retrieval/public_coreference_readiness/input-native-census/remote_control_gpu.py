"""Small bounded MCP deploy/start/collect helper; no credentials printed or persisted."""
import pathlib,sys,json,hashlib,base64,gzip,shlex,argparse,datetime
P=pathlib.Path(__file__).parent;R=P.parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
rr=json.loads((R/'download-plan.json').read_text())['remote'];remote=rr['cwd']+'/input-native-census';c=Client();c.initialize();a=argparse.ArgumentParser();a.add_argument('action',choices=['deploy','start','collect']);a.add_argument('--mode',choices=['probe','full48']);a.add_argument('--owner-commit');args=a.parse_args()
def call(n,**kw):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':rr['mount_id'],**kw}});assert not r.get('isError'),r;return r.get('structuredContent') or json.loads(r['content'][0]['text'])
def command(code):
 r=call('run_command',command=shlex.quote(rr['python'])+' -c '+shlex.quote(code),cwd=remote,env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},timeout_seconds=20);assert r['exit_code']==0,r;return json.loads(r['stdout'])
if args.action=='deploy':
 names=list(json.loads((P/'gpu-plan.json').read_text())['source_sha256'])+['gpu-plan.json'];expected={}
 for n in names:
  b=(P/n).read_bytes();expected[n]=hashlib.sha256(b).hexdigest();call('write_file',path=remote+'/'+n,content=base64.b64encode(b).decode(),encoding='base64',create_parents=True)
 for n in ['runtime-dependency-manifest.json','runtime-repair-dependency-manifest.json']:
  call('write_file',path=rr['cwd']+'/'+n,content=(R/n).read_text(),create_parents=True)
 packed=gzip.compress((P/'input-only48.jsonl').read_bytes());parts=[packed[i:i+240000] for i in range(0,len(packed),240000)]
 for i,b in enumerate(parts):call('write_file',path=remote+f'/input-transfer/{i:03d}.part',content=base64.b64encode(b).decode(),encoding='base64',create_parents=True)
 code="import pathlib,hashlib,gzip,json;P=pathlib.Path('.');expected="+repr(expected)+";actual={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in expected};assert actual==expected;blob=b''.join((P/'input-transfer'/f'{i:03d}.part').read_bytes() for i in range("+str(len(parts))+"));assert hashlib.sha256(blob).hexdigest()=="+repr(hashlib.sha256(packed).hexdigest())+";b=gzip.decompress(blob);expected_input=json.loads((P/'input-manifest.json').read_text())['input_file'];assert len(b)==expected_input['bytes'] and hashlib.sha256(b).hexdigest()==expected_input['sha256'];(P/'input-only48.jsonl').write_bytes(b);print(json.dumps({'source_files':actual,'input':expected_input}))"
 d=command(code);(P/'gpu-deployment.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
elif args.action=='start':
 assert args.mode=='full48' and args.owner_commit and len(args.owner_commit)==40
 assert not (P/(args.mode+'-job.json')).exists(),'never repeat or overwrite a mode'
 if args.mode=='full48':assert json.loads((P/'probe/state.json').read_text())['status']=='complete_input_only_native_census'
 # Fresh resource check immediately before starting only the allocated UUID.
 selected_uuid='GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539'
 code="import subprocess,json,datetime;selected="+repr(selected_uuid)+";lines=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();rows=[x.split(', ') for x in lines];gpu=next(x for x in rows if x[0]==selected);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True).splitlines();owned=[x for x in apps if x.startswith(selected+',')];assert gpu[1]=='NVIDIA A30' and int(gpu[2])==0 and int(gpu[3])==0 and not owned,(gpu,owned);print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':gpu,'compute_apps_for_uuid':owned}))"
 fresh=command(code);(P/'gpu-fresh-before-full48.json').write_text(json.dumps(fresh,indent=2)+'\n')
 env=json.loads((R/'cpu-attempt2-job.json').read_text())['env'];env['CUDA_VISIBLE_DEVICES']=selected_uuid;cap=600
 command("import pathlib,json;P=pathlib.Path('.');o=P/"+repr(args.mode)+";assert not o.exists(),'refuse overwrite';o.mkdir();print(json.dumps({'ready':True}))")
 cmd='timeout --signal=TERM --kill-after=5s '+str(cap)+'s '+shlex.quote(rr['python'])+' -u native_census_gpu.py --mode '+args.mode+' >'+args.mode+'/stdout.log 2>'+args.mode+'/stderr.log'
 job=call('start_process',command=cmd,cwd=remote,env=env);started=datetime.datetime.fromisoformat(job['started_at'].replace('Z','+00:00'));record={'owner_backup_commit':args.owner_commit,'mode':args.mode,'cap_seconds':cap,'deadline_utc':(started+datetime.timedelta(seconds=cap)).isoformat(),'command':cmd,'env':env,'plan_sha256':hashlib.sha256((P/'gpu-plan.json').read_bytes()).hexdigest(),'entry_sha256':hashlib.sha256((P/'native_census_gpu.py').read_bytes()).hexdigest(),'result':job};(P/(args.mode+'-job.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
else:
 assert args.mode
 code="import pathlib,json,hashlib,gzip;P=pathlib.Path("+repr(args.mode)+");out={};\nfor n in ['state.json','stdout.log','stderr.log']:\n f=P/n\n if f.exists():\n  b=f.read_bytes();assert len(b)<800000;out[n]={'content':b.decode(),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}\nf=P/'documents.jsonl'\nif f.exists():\n b=f.read_bytes();z=gzip.compress(b);(P/'documents.jsonl.gz').write_bytes(z);out['documents']={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'gz_bytes':len(z),'gz_sha256':hashlib.sha256(z).hexdigest()}\nprint(json.dumps(out))"
 d=command(code);out=P/args.mode;out.mkdir(exist_ok=True)
 for n,v in d.items():
  if n=='documents':continue
  b=v.pop('content').encode();assert hashlib.sha256(b).hexdigest()==v['sha256'];(out/n).write_bytes(b)
 if 'documents' in d:
  pieces=[]
  for offset in range(0,d['documents']['gz_bytes'],200000):
   code="import pathlib,base64,json;f=pathlib.Path("+repr(args.mode+'/documents.jsonl.gz')+");print(json.dumps({'b64':base64.b64encode(f.read_bytes()["+str(offset)+":"+str(offset+200000)+"]).decode()}))";pieces.append(base64.b64decode(command(code)['b64']))
  z=b''.join(pieces);assert hashlib.sha256(z).hexdigest()==d['documents']['gz_sha256'];b=gzip.decompress(z);assert hashlib.sha256(b).hexdigest()==d['documents']['sha256'];(out/'documents.jsonl.gz').write_bytes(z);(out/'documents.jsonl').write_bytes(b)
 job=json.loads((P/(args.mode+'-job.json')).read_text());procs=call('list_processes');procs=procs if isinstance(procs,list) else procs.get('processes',[]);d['owned_process']=[p for p in procs if p['id']==job['result']['id']];(out/'collection.json').write_text(json.dumps(d,indent=2)+'\n');state=json.loads((out/'state.json').read_text()) if (out/'state.json').exists() else {};print(json.dumps({'state':state,'owned_process':d['owned_process']}))
