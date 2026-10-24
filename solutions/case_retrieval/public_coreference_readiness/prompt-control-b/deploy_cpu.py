"""Transfer small owned inputs and run cached CPU tokenizer only on A800 host."""
import pathlib,json,sys,hashlib,gzip,base64,shlex
P=pathlib.Path(__file__).parent;sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
cfg=json.loads((P/'cpu-plan.json').read_text());c=Client();c.initialize();remote=cfg['cwd']+'/'+cfg['remote_root']
def call(n,**kw):
 r=c.request('tools/call',{'name':n,'arguments':{'mount_id':cfg['mount_id'],**kw}});assert not r.get('isError');return r.get('structuredContent') or json.loads(r['content'][0]['text'])
for n in ['token_audit.py','cpu-plan.json','old-actual-tokens.json']:
 call('write_file',path=remote+'/'+n,content=(P/n).read_text(),create_parents=True)
for n in ['prompts.jsonl','old-prompts.jsonl']:
 b=gzip.compress((P/n).read_bytes());parts=[b[i:i+240000] for i in range(0,len(b),240000)]
 for i,v in enumerate(parts):call('write_file',path=remote+f'/transfer-{n}/{i:03d}.part',content=base64.b64encode(v).decode(),encoding='base64',create_parents=True)
 code="import pathlib,gzip,hashlib,json;P=pathlib.Path('.');b=b''.join((P/"+repr('transfer-'+n)+"/f'{i:03d}.part').read_bytes() for i in range("+str(len(parts))+"));assert hashlib.sha256(b).hexdigest()=="+repr(hashlib.sha256(b).hexdigest())+";raw=gzip.decompress(b);assert hashlib.sha256(raw).hexdigest()=="+repr(hashlib.sha256((P/n).read_bytes()).hexdigest())+";(P/"+repr(n)+").write_bytes(raw);print(json.dumps({'verified':True}))"
 x=call('run_command',command=shlex.quote(cfg['python'])+' -c '+shlex.quote(code),cwd=remote,env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},timeout_seconds=15);assert x['exit_code']==0
x=call('run_command',command='timeout --signal=TERM --kill-after=5s 60s '+shlex.quote(cfg['python'])+' token_audit.py',cwd=remote,env=cfg['cpu_env'],timeout_seconds=60);(P/'token-audit-execution.json').write_text(json.dumps(x,indent=2)+'\n');assert x['exit_code']==0,x
r=call('read_file',path=remote+'/token-audit.json',offset=0,limit=500000);assert hashlib.sha256(r['content'].encode()).hexdigest()==r['sha256'];(P/'token-audit.json').write_text(r['content']);print(x['stdout'])
