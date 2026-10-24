"""Read-only HTTP allowlist for one immutable complete epoch and its run manifest."""
from pathlib import Path, PurePosixPath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse, hashlib, json, re


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
    cp=Path(a.checkpoint).resolve();root=cp.parent
    assert str(root)=='/dev/shm/patientphex-medgemma15-1045-20261024/fullcontext-formal2ep' and root.resolve()==root
    meta=json.loads((cp/'checkpoint.json').read_text())
    assert cp.name==f"epoch-{meta['completed_epochs']:03d}" and meta['completed_epochs'] in (1,2)
    assert meta['global_step']==16*meta['completed_epochs'] and not meta['smoke_only']
    names={}
    for n,digest in meta['files'].items():
        rel=PurePosixPath(n);assert not rel.is_absolute() and '..' not in rel.parts
        path=cp/n;assert path.is_file() and not path.is_symlink() and sha(path)==digest
        names[n]=path
    assert {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}<=set(names)
    names['checkpoint.json']=cp/'checkpoint.json'
    for n in ['run_manifest.json','train_medgemma32_fullcontext.py','train_medgemma32.py','medgemma_interface.py','qwen35_smoke.py','qwen35_relation_cost.py','fullcontext_training_contract.py','fullcontext_contract.py','construct.py','legacy_full_context.py','progress.json','fullcontext-cpu-reference-reuse.json','token-mask-audit.json','inference-input-audit.json','training-schedule.json']:
        path=root/n;assert path.is_file() and not path.is_symlink();names['_run/'+n]=path
    cpu=Path('research/medgemma-relation-control/fullcontext-probe-a800-v1').resolve()
    for n in ['reference/summary.json','reference/row-token-audit.json','preparation.json']:
        path=cpu/n;assert path.is_file() and not path.is_symlink();names['_cpu/'+n]=path
    files={n:{'bytes':q.stat().st_size,'sha256':sha(q)} for n,q in sorted(names.items())}
    manifest={'source_checkpoint_directory':str(cp),'completed_epochs':meta['completed_epochs'],
              'global_step':meta['global_step'],'checkpoint_sha256':files['checkpoint.json']['sha256'],
              'run_manifest_sha256':files['_run/run_manifest.json']['sha256'],'files':files}
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
    server=ThreadingHTTPServer(('0.0.0.0',0),Handler)
    print(json.dumps({'ready':True,'port':server.server_address[1],
        'manifest_sha256':hashlib.sha256(payload).hexdigest(),'manifest':manifest}),flush=True)
    server.serve_forever()


if __name__=='__main__':main()
