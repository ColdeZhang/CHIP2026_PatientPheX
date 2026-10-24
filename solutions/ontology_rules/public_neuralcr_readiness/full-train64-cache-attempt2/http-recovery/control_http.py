from pathlib import Path
import argparse,base64,hashlib,json,shlex,sys,time
R=Path(__file__).resolve().parent;A='476f81cf-09e7-445b-a7f0-d0e12b12e50f';P='5a9955da-5240-41cc-9e52-05d3d8da8a3c';REMOTE='public-neuralcr-readiness/full-train64-cache-attempt2/http-recovery';DEST='public-neuralcr-readiness/persistent-recovery-v1/artifacts';FETCH='public-neuralcr-readiness/persistent-recovery-v1/tools/fetch_cache.py';plan=json.loads((R/'export-plan.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
def save(n,v):
 with (R/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
p=argparse.ArgumentParser();p.add_argument('action',choices=['deploy','start','ready','fetch','collect','stop']);a=p.parse_args();sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize()
def call(m,n,args):
 x=c.request('tools/call',{'name':n,'arguments':{'mount_id':m,**args}})
 if x.get('isError'):raise RuntimeError(str(x.get('content'))[:700])
 return x['structuredContent']
if a.action=='deploy':
 items=[(A,'serve_cache.py',REMOTE+'/serve_cache.py'),(A,'export-plan.json',REMOTE+'/export-plan.json'),(P,'fetch_cache.py',FETCH)];save('deployment-intent.json',[{'mount':m,'path':rp,'sha256':sha((R/n).read_bytes())} for m,n,rp in items]);verified=[]
 for m,n,rp in items:
  b=(R/n).read_bytes();x=call(m,'write_file',{'path':rp,'content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True});assert x['sha256']==sha(b) and x['size']==len(b);verified.append({'mount':m,'path':rp,'bytes':len(b),'sha256':sha(b)})
 save('deployment.json',{'status':'all_three_sources_actual_write_SHA_verified','files':verified});print('HTTP source/plan and persistent fetcher deployed');raise SystemExit()
if a.action=='start':
 assert (R/'deployment.json').exists() and not(R/'server-start-intent.json').exists();req={'command':'timeout --signal=TERM --kill-after=5s 240s python3 -u serve_cache.py --plan export-plan.json','cwd':REMOTE,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'}};save('server-start-intent.json',req);x=call(A,'start_process',req);save('server-job.json',x);print(json.dumps(x));raise SystemExit()
if a.action=='ready':
 job=json.loads((R/'server-job.json').read_text());x=call(A,'read_process_output',{'process_id':job['id'],'stream':'stdout','cursor':0,'limit':30000});ready=[json.loads(l) for l in x['content'].splitlines() if l.startswith('{')];assert len(ready)==1 and ready[0]['ready'] and ready[0]['bind_address']==plan['bind_address'];r=ready[0];assert r['manifest']['source_files']==plan['files'];save('server-ready.json',r);print(json.dumps({'ready':True,'port':r['port'],'manifest_sha256':r['manifest_sha256']}));raise SystemExit()
if a.action=='fetch':
 r=json.loads((R/'server-ready.json').read_text());assert not(R/'fetch-start-intent.json').exists();argv=['timeout','--signal=TERM','--kill-after=5s','90s','python3','-u',FETCH,'--url','http://'+r['bind_address']+':'+str(r['port']),'--manifest-sha256',r['manifest_sha256'],'--destination',DEST,'--deadline',str(time.time()+90)];req={'command':shlex.join(argv),'cwd':'.','env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONDONTWRITEBYTECODE':'1'}};save('fetch-start-intent.json',req);x=call(P,'start_process',req);save('fetch-job.json',x);print(json.dumps(x));raise SystemExit()
if a.action=='collect':
 job=json.loads((R/'fetch-job.json').read_text());x=call(P,'read_process_output',{'process_id':job['id'],'stream':'stdout','cursor':0,'limit':30000});assert not x['running'],'fetch still running; read known job later, never repeat start';save('fetch-terminal.json',x)
 code="from pathlib import Path;import hashlib,json;p=Path("+repr(DEST)+");a=json.loads((p/'export-ack.json').read_text());assert a['status']=='persistent_cache_artifacts_verified';assert all((p/n).is_file() and not (p/n).is_symlink() and (p/n).stat().st_size==r['bytes'] and hashlib.sha256((p/n).read_bytes()).hexdigest()==r['sha256'] for n,r in a['files'].items());print(json.dumps(a))"
 req={'command':'python3 -c '+shlex.quote(code),'cwd':'.','env':{'CUDA_VISIBLE_DEVICES':''},'timeout_seconds':20};save('persistent-readback-intent.json',req);actual=call(P,'run_command',req);save('persistent-readback.json',actual);assert actual['exit_code']==0;ack=json.loads(actual['stdout']);r=json.loads((R/'server-ready.json').read_text());assert ack['files']==r['manifest']['files'] and ack['source_files']==plan['files'] and ack['transport_manifest_sha256']==r['manifest_sha256'];save('export-ack.json',ack);print(json.dumps({'status':ack['status'],'bytes':ack['total_bytes'],'seconds':ack['seconds']}));raise SystemExit()
job=json.loads((R/'server-job.json').read_text());save('server-stop-intent.json',{'process_id':job['id']});x=call(A,'stop_process',{'process_id':job['id']});save('server-stopped.json',x);print(json.dumps(x))
