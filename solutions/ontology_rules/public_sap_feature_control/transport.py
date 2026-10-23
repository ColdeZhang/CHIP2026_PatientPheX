"""Deploy/start/collect only this bounded experiment through the existing MCP."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import zipfile


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);ap.add_argument('action',choices=['deploy','start','collect']);ap.add_argument('--stage',choices=['prepare','encode','cosine']);a=ap.parse_args()
    r=Path(a.directory);plan=json.loads((r/'plan.json').read_text());runtime=plan['runtime'];mount=runtime['mount_id'];cwd=plan['remote_cwd']
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
    from mcp_client import Client
    c=Client();c.initialize()
    def call(name,args):
        result=c.request('tools/call',{'name':name,'arguments':dict(mount_id=mount,**args)})
        assert not result.get('isError'),result
        return result.get('structuredContent') or json.loads(result['content'][0]['text'])
    def save(name,value):
        with (r/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    if a.action=='deploy':
        source=Path(__file__).with_name('vector_pipeline.py');assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['vector_source_sha256']
        data={'vector_pipeline.py':source.read_bytes(),'plan.json':(r/'plan.json').read_bytes(),'request.json.gz':(r/'request.json.gz').read_bytes()}
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
            for name,b in data.items():z.writestr(name,b)
        content=buffer.getvalue()
        for index,offset in enumerate(range(0,len(content),196608)):
            call('write_file',dict(path=cwd+'/upload/part-%03d'%index,content=base64.b64encode(content[offset:offset+196608]).decode(),encoding='base64',create_parents=True))
        code="from pathlib import Path;import zipfile,io,hashlib,json;b=b''.join(p.read_bytes() for p in sorted(Path('upload').glob('part-*')));assert hashlib.sha256(b).hexdigest()=="+repr(hashlib.sha256(content).hexdigest())+";assert not Path('vector_pipeline.py').exists();z=zipfile.ZipFile(io.BytesIO(b));z.extractall('.');print(json.dumps({n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in z.namelist()}))"
        result=call('run_command',dict(command='/usr/bin/python3 -c '+shlex.quote(code),cwd=cwd,env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30));assert result['exit_code']==0
        actual=json.loads(result['stdout']);assert all(actual[n]==hashlib.sha256(b).hexdigest() for n,b in data.items())
        save('deployment.json',dict(zip_bytes=len(content),files=actual))
    elif a.action=='start':
        assert a.stage is not None and not (r/(a.stage+'-job.json')).exists()
        deployed=json.loads((r/'deployment.json').read_text())
        assert deployed['files']['plan.json']==hashlib.sha256((r/'plan.json').read_bytes()).hexdigest()
        assert deployed['files']['vector_pipeline.py']==plan['vector_source_sha256']
        previous={'encode':'prepare','cosine':'encode'}.get(a.stage)
        if previous is not None:assert (r/(previous+'-transfer.json')).exists()
        seconds=60 if a.stage=='encode' else 120
        env=dict(runtime['environment'],CUDA_VISIBLE_DEVICES=plan['GPU_UUID'] if a.stage=='encode' else '')
        command='timeout --signal=TERM --kill-after=5s %ds '%seconds+runtime['python']+' vector_pipeline.py '+a.stage
        job=call('start_process',dict(command=command,cwd=cwd,env=env))
        save(a.stage+'-job.json',dict(command=command,environment=env,plan_sha256=hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),result=job));print(json.dumps(job))
    else:
        assert a.stage is not None
        job=json.loads((r/(a.stage+'-job.json')).read_text())
        receipt={s:call('read_process_output',dict(process_id=job['result']['id'],stream=s,cursor=0,limit=32768)) for s in ['stdout','stderr']}
        assert all(not x['running'] for x in receipt.values()),'job still running'
        save(a.stage+'-managed-terminal.json',receipt)
        files={'prepare':['prepared/manifest.json','prepared/missing.json','prepare-complete.json'],
               'encode':['new-embeddings/manifest.json','new-embeddings/fresh-GPU.json','encode-complete.json'],
               'cosine':['features/lookup.jsonl.gz','features/manifest.json','cosine-complete.json']}[a.stage]
        name=a.stage+'-terminal.zip'
        code="from pathlib import Path;import json,zipfile,hashlib;files="+repr(files)+";z=zipfile.ZipFile("+repr(name)+",'x',zipfile.ZIP_DEFLATED);[z.write(p,p) for p in files];z.close();p=Path("+repr(name)+");print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'files':{n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in files}}))"
        result=call('run_command',dict(command='/usr/bin/python3 -c '+shlex.quote(code),cwd=cwd,env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=30));assert result['exit_code']==0,result
        manifest=json.loads(result['stdout']);buffer=io.BytesIO()
        for offset in range(0,manifest['bytes'],262144):
            chunk=call('read_file',dict(path=cwd+'/'+name,offset=offset,limit=min(262144,manifest['bytes']-offset)))
            assert chunk['encoding']=='base64';buffer.write(base64.b64decode(chunk['content']))
        data=buffer.getvalue();assert hashlib.sha256(data).hexdigest()==manifest['sha256']
        target=r/'terminal';target.mkdir(exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for file,digest in manifest['files'].items():
                b=z.read(file);assert hashlib.sha256(b).hexdigest()==digest
                p=target/file;p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(b)
        save(a.stage+'-transfer.json',manifest);print(json.dumps(manifest))


if __name__=='__main__':main()
