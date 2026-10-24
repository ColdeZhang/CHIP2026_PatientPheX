"""Read-only fixed NCR artifact allowlist; HTTP Range handler reused unchanged."""
from pathlib import Path, PurePosixPath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse,hashlib,ipaddress,json,os,re

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();plan=json.loads(Path(a.plan).read_text());address=ipaddress.ip_address(plan['bind_address']);assert address.is_private and not address.is_loopback and address.version==4
    root=Path(__file__).resolve().parent;assert not(root/'server-ready.json').exists();names={};files={}
    assert set(plan['files'])=={'native-raw-spans.jsonl.gz','columns.json','top1.npy.gz','batch-progress.jsonl','finalization-result.json'}
    for name,r in plan['files'].items():
        assert PurePosixPath(name).name==name;path=Path(r['path']);assert str(path).startswith('/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-neuralcr-readiness/') and path.is_file() and not path.is_symlink() and path.stat().st_size==r['bytes'] and sha(path)==r['sha256'];names[name]=path;files[name]={'bytes':r['bytes'],'sha256':r['sha256']}
    manifest={'source_cache_directory':plan['source_cache_directory'],'source_files':plan['files'],'plan_sha256':sha(a.plan),'source_sha256':sha(__file__),'files':files,'scope':'Read-only exact complete NCR artifacts; original failed run and CPU finalization retained.'}
    payload=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/manifest.json':
                self.send_response(200);self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload);return
            prefix='/files/';name=self.path[len(prefix):] if self.path.startswith(prefix) else ''
            if name not in names or self.path!=prefix+name:self.send_error(404);return
            path=names[name];size=files[name]['bytes'];start=0;end=size-1;status=200
            if self.headers.get('Range'):
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
                if not match:self.send_error(416);return
                start=int(match[1]);end=int(match[2]) if match[2] else size-1
                if not 0<=start<=end<size:self.send_error(416);return
                status=206
            self.send_response(status);self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
            if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            with path.open('rb') as f:
                f.seek(start);left=end-start+1
                while left:
                    chunk=f.read(min(left,1048576))
                    if not chunk:raise OSError('Frozen checkpoint ended before declared size')
                    self.wfile.write(chunk);left-=len(chunk)
    server=ThreadingHTTPServer((plan['bind_address'],0),Handler)
    ready={'ready':True,'pid':os.getpid(),'bind_address':plan['bind_address'],'port':server.server_address[1],'manifest_sha256':hashlib.sha256(payload).hexdigest(),'manifest':manifest}
    with (root/'server-ready.json').open('x') as f:json.dump(ready,f,indent=2);f.write('\n')
    print(json.dumps(ready),flush=True);server.serve_forever()

if __name__=='__main__':main()
