#!/usr/bin/env python3
"""Only two pinned SFT32 files via temporary allowlist server; never base weights."""
from pathlib import Path
import argparse,hashlib,json,shlex,sys,time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory
    p=json.loads((b/'plan.json').read_text());assert not (b/'adapter-transfer.json').exists() and not (b/'adapter-transfer-job.json').exists()
    sys.path.insert(0,'/workspace/patientphex-tools');from mcp_client import Client
    c=Client();c.initialize()
    def call(mount,name,args):
        r=c.request('tools/call',dict(name=name,arguments=dict(mount_id=mount,**args)));assert not r.get('isError'),r
        return r['structuredContent']
    def save(name,x):
        with (b/name).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
    src=p['adapter_transfer'];server_root='sequence-round2/retrieved-json-example-transfer'
    # The existing adapter is read-only; server scripts/logs use this own directory.
    server='''from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import hashlib,json
ROOT=Path(ROOT_PATH);ALLOW=ALLOWED
for n,h in ALLOW.items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):
  name=self.path.removeprefix('/'+PREFIX+'/')
  if self.path!='/'+PREFIX+'/'+name or name not in ALLOW:self.send_error(404);return
  path=ROOT/name;self.send_response(200);self.send_header('Content-Length',str(path.stat().st_size));self.end_headers()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(1048576),b''):self.wfile.write(block)
print('allowlist ready',flush=True)
ThreadingHTTPServer(('0.0.0.0',PORT),Handler).serve_forever()
'''.replace('ROOT_PATH',repr(src['source_directory'])).replace('ALLOWED',repr(p['adapter_files'])).replace('PREFIX',repr(src['path_prefix'])).replace('PORT',str(src['port']))
    mkdir=call(src['mount'],'run_command',dict(command='python3 -c '+shlex.quote(
        'from pathlib import Path;p=Path('+repr(server_root)+');assert not p.exists();p.mkdir(parents=True)'),cwd='',timeout_seconds=10))
    assert mkdir['exit_code']==0
    save('adapter-server-directory.json',mkdir)
    call(src['mount'],'write_file',dict(path=server_root+'/serve.py',encoding='utf8',content=server))
    started=call(src['mount'],'start_process',dict(command='timeout --signal=TERM --kill-after=5s 360s python3 serve.py',cwd=server_root))
    save('adapter-server-job.json',dict(result=started,source_sha256=hashlib.sha256(server.encode()).hexdigest(),allowed=p['adapter_files']))
    # A few bounded ready polls; never start a second server.
    for _ in range(10):
        ready=call(src['mount'],'read_process_output',dict(process_id=started['id'],stream='stdout',cursor=0,limit=2000))
        if 'allowlist ready' in ready.get('content',''):break
        time.sleep(.5)
    else:raise RuntimeError('Allowlist server not ready')
    downloader='''from pathlib import Path
import urllib.request,hashlib,json,time,datetime
started=time.monotonic();target=Path(TARGET);target.mkdir(parents=True,exist_ok=False);allow=ALLOWED;items={}
for name,digest in allow.items():
 path=target/name;h=hashlib.sha256();size=0
 with urllib.request.urlopen(BASE_URL+name,timeout=25) as response,path.open('xb') as f:
  while True:
   block=response.read(1048576)
   if not block:break
   f.write(block);h.update(block);size+=len(block)
 assert h.hexdigest()==digest
 items[name]=dict(bytes=size,sha256=digest)
out=dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=items,seconds=time.monotonic()-started)
(target/'transfer.json').write_text(json.dumps(out)+'\\n');print(json.dumps(out),flush=True)
'''.replace('TARGET',repr(p['adapter'])).replace('ALLOWED',repr(p['adapter_files'])).replace('BASE_URL',repr('http://'+src['host']+':'+str(src['port'])+'/'+src['path_prefix']+'/'))
    save('adapter-transfer-source.json',dict(server_source=server,downloader_source=downloader))
    job=call(p['mount'],'start_process',dict(command='timeout --signal=TERM --kill-after=5s 300s python3 -c '+shlex.quote(downloader),cwd=''))
    save('adapter-transfer-job.json',job)
    try:
        while True:
            result=call(p['mount'],'read_process_output',dict(process_id=job['id'],stream='stdout',cursor=0,limit=4000))
            if not result['running']:break
            time.sleep(5)
        err=call(p['mount'],'read_process_output',dict(process_id=job['id'],stream='stderr',cursor=0,limit=4000))
        save('adapter-transfer-process.json',dict(stdout=result,stderr=err))
        check=call(p['mount'],'run_command',dict(command='python3 -c '+shlex.quote('from pathlib import Path;import hashlib,json;p=Path('+repr(p['adapter'])+');x=json.loads((p/"transfer.json").read_text());assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==r["sha256"] for n,r in x["files"].items());print(json.dumps(x))'),cwd='',timeout_seconds=15))
        assert check['exit_code']==0
        save('adapter-transfer.json',dict(actual_transfer=json.loads(check['stdout']),readback=check))
    finally:
        stop=call(src['mount'],'stop_process',dict(process_id=started['id']));save('adapter-server-stopped.json',stop)
    print('Two-file adapter transfer complete; source server stopped')

if __name__=='__main__':main()
