"""Read-only fixed R1 assets, inherited verified Range/EOF transport handler."""
from pathlib import Path, PurePosixPath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse, hashlib, json, re, os


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source-manifest',required=True);p.add_argument('--bind-private-ip',required=True);a=p.parse_args()
    cp=Path(a.model).resolve()
    assert cp.name=='DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6' and cp.parent.name=='models' and os.getuid()==cp.stat().st_uid==1006
    assert a.bind_private_ip=='10.5.2.100'
    assert sha(a.source_manifest)=='ee707db3a5b314c94c213386c2c8b8640fd9406759e68f9e729640f70ccbfaec'
    expected=json.loads(Path(a.source_manifest).read_text())
    assert expected['model_id']=='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B' and expected['revision']=='5ee5fa4832e658ab84298806de06ed2a167834bd'
    assert len(expected['files'])==9
    names={};files={}
    for r in expected['files']:
        name=r['name'];assert Path(name).name==name
        path=cp/name;assert path.is_file() and not path.is_symlink() and path.stat().st_size==r['bytes'] and sha(path)==r['sha256']
        names[name]=path;files[name]={'bytes':r['bytes'],'sha256':r['sha256']}
    manifest={'source_model_directory':str(cp),'model_id':expected['model_id'],'MS_revision':expected['revision'],
              'expected_files_sha256':sha(a.source_manifest),'files':files}
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
    server=ThreadingHTTPServer((a.bind_private_ip,0),Handler)
    print(json.dumps({'ready':True,'port':server.server_address[1],
        'manifest_sha256':hashlib.sha256(payload).hexdigest(),'manifest':manifest}),flush=True)
    server.serve_forever()


if __name__=='__main__':main()
