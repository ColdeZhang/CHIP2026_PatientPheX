"""One CPU preparation job on existing exact native environment; no GPU/model."""
import base64,hashlib,io,json,shlex,sys,zipfile,datetime,os
from pathlib import Path
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
S=Path(__file__).parent;D=Path('/workspace/patientphex-runs/round2-20261023/research/public-h5-train32-preparation');M='476f81cf-09e7-445b-a7f0-d0e12b12e50f';REMOTE='public-h5-train32-cpu-v1';N='/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse';c=Client();c.initialize()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def call(n,a):
 r=c.request('tools/call',dict(name=n,arguments=dict(mount_id=M,**a)));assert not r.get('isError');return r['structuredContent']
def command(code):
 r=call('run_command',dict(command=N+'/.venv/bin/python -c '+shlex.quote(code),timeout_seconds=30));assert r['exit_code']==0,r.get('stderr','')[-800:];return json.loads(r['stdout'])
mode=sys.argv[1]
if mode=='prepare':
 code="""from pathlib import Path
import importlib.metadata as md,json,hashlib,sys
root=Path(NATIVE);paths={'native_representation':root/'nn_represent.py','native_ml':root/'native-pipeline/src/ml_ner.py','native_splitter':root/'native-pipeline/src/ssplit_tokenzier.py','native_restore':root/'native-pipeline/src/restore_index.py','fixed_candidate_builder':root/'native-pipeline/fixed_candidate_function.py','label_vocab':root/'lable.vocab'}
for n in ['config.json','vocab.txt','tokenizer_config.json']:paths['tokenizer_'+n]=root/'pubmedbert-base'/n
print(json.dumps(dict(python=sys.executable,packages={x:md.version(x) for x in ['transformers','tokenizers','nltk','tensorflow-cpu','numpy']},assets={k:dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in paths.items()},new_directory_exists=Path(REMOTE).exists())))
""".replace('NATIVE',repr(N)).replace('REMOTE',repr(REMOTE))
 preflight=D/'remote-preflight-revision2.json'
 if preflight.exists():actual=json.loads(preflight.read_text())
 else:actual=command(code);write(preflight,actual)
 assert not actual['new_directory_exists'];inputs=json.loads((S/'inputs.json').read_text())['refs'];equivalence={}
 for k,r in actual['assets'].items():
  local=Path(inputs[k]['path']).read_bytes();exact=hashlib.sha256(local).hexdigest()==r['sha256'] and len(local)==r['bytes']
  lf=local.replace(b'\r\n',b'\n');normalized=hashlib.sha256(lf).hexdigest()==r['sha256'] and len(lf)==r['bytes']
  assert exact or (k in ['native_representation','native_ml'] and normalized),k
  equivalence[k]=dict(local=inputs[k],actual=r,byte_exact=exact,only_CRLF_to_LF=(not exact and normalized))
 write(D/'asset-equivalence.json',equivalence)
 entry=dict(native_root=N,output_directory='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+REMOTE+'/native-result',assets=actual['assets'],request=ref(D/'local/request.json.gz'),CPU_only=True,source_refs={n:ref(S/n) for n in ['native_cpu.py','loss_contract.py','cpu_driver.py']})
 entry['request']['path']='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+REMOTE+'/request.json.gz';write(D/'entry.json',entry);write(S/'entry.json',entry);print(json.dumps(dict(status='CPU_ENTRY_BOUND',assets_equal=True,packages=actual['packages'],entry=ref(D/'entry.json'))))
elif mode=='deploy':
 files={n:S/n for n in ['native_cpu.py','loss_contract.py','cpu_driver.py']};files['entry.json']=D/'entry.json';files['request.json.gz']=D/'local/request.json.gz'
 buf=io.BytesIO()
 with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
  for n,p in files.items():z.writestr(n,p.read_bytes())
 b=buf.getvalue();h=hashlib.sha256(b).hexdigest();write(D/'deploy-intent.json',dict(archive_sha256=h,files={n:ref(p) for n,p in files.items()}))
 for i,off in enumerate(range(0,len(b),196608)):call('write_file',dict(path=REMOTE+'-upload/part-%03d'%i,content=base64.b64encode(b[off:off+196608]).decode(),encoding='base64',create_parents=True))
 code="""from pathlib import Path
import hashlib,io,json,zipfile,os
root=Path(REMOTE);assert not root.exists();parts=sorted(Path(REMOTE+'-upload').glob('part-*'));b=b''.join(p.read_bytes() for p in parts);assert hashlib.sha256(b).hexdigest()==HASH;root.mkdir();z=zipfile.ZipFile(io.BytesIO(b));assert all('/' not in n for n in z.namelist());z.extractall(root);out={}
for p in root.iterdir():
 with p.open('rb') as f:os.fsync(f.fileno())
 out[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(out))
""".replace('REMOTE',repr(REMOTE)).replace('HASH',repr(h));actual=command(code)
 for n,p in files.items():assert actual[n]['sha256']==sha(p) and actual[n]['bytes']==p.stat().st_size
 write(D/'deployment.json',dict(archive_sha256=h,files=actual));print(json.dumps(dict(status='DEPLOYED_CPU_ONLY',bytes=len(b))))
elif mode=='start':
 assert (D/'deployment.json').exists();write(D/'start-intent.json',dict(entry=ref(D/'entry.json'),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True));job=call('start_process',dict(command=N+'/.venv/bin/python cpu_driver.py',cwd=REMOTE,env=dict(CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='1',TF_NUM_INTEROP_THREADS='1',TF_NUM_INTRAOP_THREADS='1')));write(D/'job.json',job);print(json.dumps(job))
elif mode=='status':
 job=json.loads((D/'job.json').read_text());m=call('read_process_output',dict(process_id=job['id'],stream='stdout',cursor=0,limit=1024));state=command("from pathlib import Path;import json;p=Path("+repr(REMOTE+'/state.json')+");print(p.read_text() if p.exists() else '{}')");(D/'latest-status.json').write_text(json.dumps(dict(managed=m,state=state),indent=2)+'\n');print(json.dumps(dict(running=m.get('running'),state=state)))
elif mode=='collect':
 manifest=command("from pathlib import Path;print(Path("+repr(REMOTE+'/terminal-manifest.json')+").read_text())");assert not manifest['state']['running'];write(D/'terminal-manifest.json',manifest);archive=manifest['archive'];chunks=[]
 for off in range(0,archive['bytes'],196608):
  x=command("from pathlib import Path;import base64,json;p=Path("+repr(archive['path'])+");f=p.open('rb');f.seek("+str(off)+");print(json.dumps(dict(data=base64.b64encode(f.read(196608)).decode())))");chunks.append(base64.b64decode(x['data']))
 b=b''.join(chunks);assert len(b)==archive['bytes'] and hashlib.sha256(b).hexdigest()==archive['sha256'];(D/'terminal.zip').write_bytes(b);out=D/'terminal';out.mkdir(exist_ok=False)
 with zipfile.ZipFile(io.BytesIO(b)) as z:
  assert all('..' not in n and not n.startswith('/') for n in z.namelist());assert json.loads(z.read('members.json'))==manifest['members'];z.extractall(out)
 for n,r in manifest['members'].items():assert sha(out/n)==r['sha256'] and (out/n).stat().st_size==r['bytes']
 write(D/'collection.json',dict(archive=archive,members=len(manifest['members']),all_reopened_SHA_verified=True));print(json.dumps(dict(returncode=manifest['state']['returncode'],members=len(manifest['members']),archive_bytes=archive['bytes'])))
else:raise ValueError(mode)
