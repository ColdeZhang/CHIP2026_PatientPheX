"""TTL-limited private peer transfer of an exact public asset archive."""
import argparse
import hashlib
import http.server
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
import urllib.request
from assets import digest, ref, write

def pack(args):
    root=Path(args.root);complete=json.loads((root/'complete.json').read_text())
    assert complete['status']=='COMPLETE_EXACT_ASSETS_AND_SAFE_SOURCE_NO_MODEL_OR_GPU'
    paths={Path(a['path']).relative_to(root).as_posix():a for a in complete['assets']}
    paths.update({n:ref(root/n) for n in ['complete.json','asset-plan.json']})
    for p in (root/'application').glob('*'):
        if p.is_file():paths[p.relative_to(root).as_posix()]=ref(p)
    for name,item in paths.items():assert ref(root/name)['sha256']==item['sha256']
    write(root/'transfer-members.json',paths)
    with tarfile.open(root/'transfer.tar','x') as tar:
        for name in sorted(paths):tar.add(root/name,arcname=name,recursive=False)
        tar.add(root/'transfer-members.json',arcname='transfer-members.json',recursive=False)
    write(root/'transfer-ready.json',dict(archive=ref(root/'transfer.tar'),members=len(paths),source=ref(Path(__file__))))

def serve(args):
    root=Path(args.root);ready=json.loads((root/'transfer-ready.json').read_text());archive=root/'transfer.tar'
    assert digest(archive)==ready['archive']['sha256'];token=Path(args.token_file).read_text();port=int(os.environ['PORT'])
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path!='/frozen.tar' or self.headers.get('Authorization')!='Bearer '+token or self.client_address[0]!=args.peer:
                self.send_error(403);return
            self.send_response(200);self.send_header('Content-Type','application/x-tar');self.send_header('Content-Length',str(archive.stat().st_size));self.end_headers()
            with archive.open('rb') as f:shutil.copyfileobj(f,self.wfile,length=1<<20)
    server=http.server.ThreadingHTTPServer((args.bind,port),Handler)
    timer=threading.Timer(args.ttl,server.shutdown);timer.start()
    write(root/'transfer-server-started.json',dict(pid=os.getpid(),port=port,bind=args.bind,peer=args.peer,ttl=args.ttl,started_unix=time.time(),archive_sha256=ready['archive']['sha256']))
    try:server.serve_forever()
    finally:timer.cancel();server.server_close();write(root/'transfer-server-ended.json',dict(ended_unix=time.time()))

def fetch(args):
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True);dest=root/'transfer.tar';part=root/'transfer.tar.part'
    token=Path(args.token_file).read_text()
    with urllib.request.urlopen(urllib.request.Request(args.url,headers={'Authorization':'Bearer '+token}),timeout=40) as response:
        with part.open('xb') as f:shutil.copyfileobj(response,f,length=1<<20);f.flush();os.fsync(f.fileno())
    assert digest(part)==args.sha256 and part.stat().st_size==args.bytes
    os.replace(part,dest)
    with tarfile.open(dest) as tar:
        members=json.load(tar.extractfile('transfer-members.json'))
        for name,item in members.items():
            relative=PurePosixPath(name);assert not relative.is_absolute() and '..' not in relative.parts
            member=tar.getmember(name);assert member.isfile();target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('xb') as output:shutil.copyfileobj(source,output,length=1<<20);output.flush();os.fsync(output.fileno())
            assert target.stat().st_size==item['bytes'] and digest(target)==item['sha256']
    # The pinned source archive contains many files. Only package+license are used.
    with tarfile.open(root/'source.tar.gz') as tar:
        for member in tar:
            parts=PurePosixPath(member.name).parts
            if not member.isfile() or len(parts)<2:continue
            relative=PurePosixPath(*parts[1:]);assert not relative.is_absolute() and '..' not in relative.parts
            if relative.parts[0]!='gliner2' and str(relative) not in ['pyproject.toml','LICENSE']:continue
            target=root/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as source,target.open('xb') as output:shutil.copyfileobj(source,output)
    write(root/'transfer-complete.json',dict(status='EXACT_SOURCE_AND_ALL_MEMBERS_VERIFIED',archive=ref(dest),members=members,completed_unix=time.time()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['pack','serve','fetch']);ap.add_argument('--root',required=True);ap.add_argument('--token-file');ap.add_argument('--bind');ap.add_argument('--peer');ap.add_argument('--ttl',type=int,default=180);ap.add_argument('--url');ap.add_argument('--sha256');ap.add_argument('--bytes',type=int);args=ap.parse_args();globals()[args.mode](args)
if __name__=='__main__':main()
