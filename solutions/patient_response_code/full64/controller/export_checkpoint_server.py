"""Read-only HTTP allowlist for one complete Qwen35 response-code epoch; no GPU/model import.
Owner bounds this separate final export to CPU600s maximum within original5400s deadline and stops this dynamic-port server.
"""
from pathlib import Path, PurePosixPath
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse, hashlib, json, re, time

CHECKPOINT_REQUIRED = {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}
CHECKPOINT_ALLOWED = CHECKPOINT_REQUIRED | {'adapter/README.md'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());SOURCE_RUN_ROOT=Path(plan['output_dir']);TRAINER_SHA=plan['trainer_sha256'];TRAIN_SHA=plan['train_sha256'];RUN_FILES=plan['export_run_files']
    assert time.time()<a.deadline
    requested=Path(a.checkpoint);assert not requested.is_symlink()
    cp=requested.resolve();root=cp.parent
    assert root==SOURCE_RUN_ROOT and cp.name in {'epoch-001','epoch-002'}
    meta=json.loads((cp/'checkpoint.json').read_text())
    assert cp.name==f"epoch-{meta['completed_epochs']:03d}" and meta['completed_epochs'] in (1,2)
    assert meta['global_step']==34*meta['completed_epochs'] and not meta['smoke_only']
    assert meta['fingerprint']['schema']==plan['fingerprint_schema']
    assert meta['fingerprint']['source_sha256']==TRAINER_SHA and meta['fingerprint']['train_sha256']==TRAIN_SHA
    assert CHECKPOINT_REQUIRED<=set(meta['files'])<=CHECKPOINT_ALLOWED
    assert sha(root/plan['export_trainer_basename'])==TRAINER_SHA
    run_meta=json.loads((root/'run_manifest.json').read_text())
    assert run_meta['fingerprint']==meta['fingerprint']
    assert (run_meta['epochs'],run_meta['total_optimizer_steps'],run_meta['warmup_steps'])==(2,68,4)
    names={}
    for n,digest in meta['files'].items():
        rel=PurePosixPath(n);assert not rel.is_absolute() and '..' not in rel.parts and str(rel)==n
        assert re.fullmatch('[0-9a-f]{64}',digest)
        path=cp/n;assert cp in path.resolve().parents
        assert path.is_file() and not path.is_symlink() and sha(path)==digest
        names[n]=path
    assert CHECKPOINT_REQUIRED<=set(names)
    names['checkpoint.json']=cp/'checkpoint.json'
    run_names=RUN_FILES+(['training_complete.json'] if meta['completed_epochs']==2 else [])
    for n in run_names:
        path=root/n;assert path.is_file() and not path.is_symlink();names['_run/'+n]=path
    if meta['completed_epochs']==2:
        completed=json.loads((root/'training_complete.json').read_text())
        assert completed['epochs']==2 and completed['global_step']==68 and not completed['smoke_only']
        assert completed['source_sha256']==TRAINER_SHA
    files={n:{'bytes':q.stat().st_size,'sha256':sha(q)} for n,q in sorted(names.items())}
    manifest={'source_checkpoint_directory':str(cp),'completed_epochs':meta['completed_epochs'],
              'global_step':meta['global_step'],'ack_basename':f"export-ack-epoch-{meta['completed_epochs']:03d}.json",
              'trainer_sha256':TRAINER_SHA,'checkpoint_sha256':files['checkpoint.json']['sha256'],
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
    assert time.time()<a.deadline
    print(json.dumps({'ready':True,'port':server.server_address[1],
        'manifest_sha256':hashlib.sha256(payload).hexdigest(),'manifest':manifest}),flush=True)
    server.timeout=min(1.0,max(.001,a.deadline-time.time()))
    try:
        while time.time()<a.deadline:server.handle_request()
    finally:server.server_close()


if __name__=='__main__':main()
