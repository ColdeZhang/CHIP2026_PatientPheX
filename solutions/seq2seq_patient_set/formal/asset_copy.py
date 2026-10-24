"""Whitelisted four-file private copy; same public base, no model or GPU import."""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import time
import urllib.request
from common import sha, ref, save, fsync_dir


def main(a):
    plan=json.loads(Path(a.plan).read_text());root=Path(a.directory);files=plan['model']
    assert set(files)=={'config.json','pytorch_model.bin','spiece.model','tokenizer.json'}
    secret=os.environ['SEQ2SEQ_TRANSFER_AUTH'];assert len(secret)>=32 and time.time()<a.deadline
    if a.mode=='serve':
        for n,e in files.items():assert all(ref(root/n)[k]==e[k] for k in ('bytes','sha256'))
        delivered=set()
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                name=self.path.removeprefix('/files/')
                if self.path!='/files/'+name or name not in files or time.time()>=a.deadline or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+secret):
                    self.send_error(403);return
                self.connection.settimeout(15);self.send_response(200);self.send_header('Content-Length',str(files[name]['bytes']));self.end_headers()
                with (root/name).open('rb') as f:
                    for block in iter(lambda:f.read(1048576),b''):
                        if time.time()>=a.deadline:raise TimeoutError('Transfer deadline')
                        self.wfile.write(block)
                delivered.add(name)
        server=HTTPServer((a.address,int(os.environ['PORT'])),Handler);server.timeout=.5
        print(json.dumps({'ready':True,'port':server.server_port,'files':files,'deadline':a.deadline}),flush=True)
        try:
            while time.time()<a.deadline and len(delivered)<4:server.handle_request()
        finally:server.server_close()
        print(json.dumps({'closed':True,'delivered':sorted(delivered)}),flush=True)
    else:
        root.mkdir(parents=True,exist_ok=False);start=time.monotonic();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for name,e in files.items():
            assert time.time()<a.deadline
            request=urllib.request.Request(os.environ['SEQ2SEQ_TRANSFER_URL']+'/files/'+name,headers={'Authorization':'Bearer '+secret})
            with opener.open(request,timeout=30) as response,(root/(name+'.part')).open('xb') as f:
                assert int(response.headers['Content-Length'])==e['bytes'];count=0
                for block in iter(lambda:response.read(1048576),b''):
                    assert time.time()<a.deadline;f.write(block);count+=len(block);assert count<=e['bytes']
                f.flush();os.fsync(f.fileno())
            pending=root/(name+'.part');assert count==e['bytes'] and sha(pending)==e['sha256'];pending.rename(root/name)
            assert sha(root/name)==e['sha256']
        fsync_dir(root);fsync_dir(root.parent)
        save(root/'copy-ACK.json',{'status':'COMPLETE_SAME_BASE_PRIVATE_COPY_FSYNC_REOPEN','files':{n:ref(root/n) for n in files},
            'seconds':time.monotonic()-start,'source_retained':True,'GPU_calls':0,'credential_or_URL_saved':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['serve','receive']);p.add_argument('--plan',required=True)
    p.add_argument('--directory',required=True);p.add_argument('--deadline',type=float,required=True);p.add_argument('--address');main(p.parse_args())
