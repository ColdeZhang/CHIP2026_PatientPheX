"""Local CPU fixture checks for the bounded checkpoint transport; no MCP/GPU."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,time,urllib.error,urllib.request,threading,http.client
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

HERE=Path(__file__).parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    began=time.monotonic();checks={}
    with tempfile.TemporaryDirectory(prefix='patientphex-order-warm32-1045-fixture-',dir='/dev/shm') as shared, tempfile.TemporaryDirectory(prefix='patientphex-export-fixture-') as durable:
        run=Path(shared)/'control';cp=run/'epoch-001';(cp/'adapter').mkdir(parents=True)
        values={'adapter/adapter_config.json':b'{"fixture":true}\n','adapter/adapter_model.safetensors':b'fixture-weight'*1000,'training_state.pt':b'fixture-state'*40000}
        for name,value in values.items():(cp/name).write_bytes(value)
        metadata={'completed_epochs':1,'global_step':16,'smoke_only':False,'files':{name:sha(cp/name) for name in values}}
        (cp/'checkpoint.json').write_text(json.dumps(metadata))
        for name in ['run_manifest.json','train_qwen35_meta32.py','qwen35_smoke.py','qwen35_relation_cost.py','progress.json']:(run/name).write_text('fixture '+name+'\n')
        proc=subprocess.Popen([sys.executable,str(HERE/'export_checkpoint_server.py'),'--checkpoint',str(cp)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        try:
            import select
            assert select.select([proc.stdout],[],[],8)[0],'Fixture server did not become ready'
            ready=json.loads(proc.stdout.readline());url='http://127.0.0.1:'+str(ready['port'])
            for suffix in ['/files/../run_manifest.json','/files/not-listed','/files/_run/unknown']:
                try:urllib.request.urlopen(url+suffix,timeout=3);raise AssertionError('Unlisted path served')
                except urllib.error.HTTPError as e:assert e.code==404
            checks['unlisted_and_traversal_paths_rejected']=True
            request=urllib.request.Request(url+'/files/training_state.pt',headers={'Range':'bytes=13-39'})
            with urllib.request.urlopen(request,timeout=3) as response:
                assert response.status==206 and response.read()==values['training_state.pt'][13:40]
                assert response.headers['Content-Range']==f"bytes 13-39/{len(values['training_state.pt'])}"
            checks['exact_range_response']=True
            rel='case-retrieval/qwen35-preparation/order-warmstart-stage/checkpoints/fixture/epoch-001'
            dst=Path(durable)/rel;dst.mkdir(parents=True)
            (dst/'training_state.pt.part').write_bytes(values['training_state.pt'][:43])
            argv=[sys.executable,str(HERE/'fetch_checkpoint.py'),'--url',url,'--manifest-sha256',ready['manifest_sha256'],'--destination',rel,'--deadline',str(time.time()+30)]
            completed=subprocess.run(argv,cwd=durable,text=True,capture_output=True,timeout=15)
            assert completed.returncode==0,completed.stderr
            ack=json.loads((dst/'export-ack.json').read_text());assert ack['fsync_all_files'] and ack['status']=='persistent_checkpoint_verified'
            assert ack['checkpoint_sha256']==sha(cp/'checkpoint.json') and ack['run_manifest_sha256']==sha(run/'run_manifest.json')
            for name,record in ack['files'].items():assert (dst/name).stat().st_size==record['bytes'] and sha(dst/name)==record['sha256']
            checks['partial_range_resume_complete_SHA_fsync_and_ack']=True
            duplicate=subprocess.run(argv,cwd=durable,text=True,capture_output=True,timeout=10)
            assert duplicate.returncode!=0 and 'already exported' in duplicate.stderr
            checks['completed_ack_not_overwritten']=True
            badrel=rel.replace('/fixture/','/corrupt/')
            bad=Path(durable)/badrel;bad.mkdir(parents=True);(bad/'training_state.pt').write_bytes(b'wrong')
            badargs=argv.copy();badargs[badargs.index('--destination')+1]=badrel
            badresult=subprocess.run(badargs,cwd=durable,text=True,capture_output=True,timeout=10)
            assert badresult.returncode!=0 and not (bad/'export-ack.json').exists()
            checks['wrong_existing_payload_rejected_no_ack']=True
            badargs=argv.copy();badargs[badargs.index('--manifest-sha256')+1]='0'*64
            badresult=subprocess.run(badargs,cwd=durable,text=True,capture_output=True,timeout=10)
            assert badresult.returncode!=0
            checks['wrong_transport_manifest_SHA_rejected']=True
            # Real early socket close once; next request must use the saved prefix.
            source_bytes={name:((run/name[5:]) if name.startswith('_run/') else (cp/name)).read_bytes() for name in ready['manifest']['files']}
            with urllib.request.urlopen(url+'/manifest.json',timeout=3) as response:manifest_bytes=response.read()
            interrupted={'sent':False,'ranges':[]}
            class InterruptOnce(BaseHTTPRequestHandler):
                def log_message(self,*args):pass
                def do_GET(self):
                    if self.path=='/manifest.json':data=manifest_bytes;start=0
                    else:
                        name=self.path[len('/files/'):];data=source_bytes[name]
                        header=self.headers.get('Range');start=int(header.split('=')[1].split('-')[0]) if header else 0
                        if name=='training_state.pt':interrupted['ranges'].append(start)
                        if name=='training_state.pt' and not interrupted['sent']:
                            interrupted['sent']=True;self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data[:123456]);self.close_connection=True;return
                    self.send_response(206 if start else 200);self.send_header('Content-Length',str(len(data)-start))
                    if start:self.send_header('Content-Range',f'bytes {start}-{len(data)-1}/{len(data)}')
                    self.end_headers();self.wfile.write(data[start:])
            proxy=ThreadingHTTPServer(('127.0.0.1',0),InterruptOnce);thread=threading.Thread(target=proxy.serve_forever,daemon=True);thread.start()
            try:
                retryargs=argv.copy();retryargs[retryargs.index('--url')+1]='http://127.0.0.1:'+str(proxy.server_address[1]);retryargs[retryargs.index('--destination')+1]=rel.replace('/fixture/','/interrupted/')
                retry=subprocess.run(retryargs,cwd=durable,text=True,capture_output=True,timeout=12)
                assert retry.returncode==0,retry.stderr
                assert interrupted['ranges']==[0,123456],interrupted
                checks['actual_interrupted_HTTP_resumes_exact_saved_prefix']=True
            finally:proxy.shutdown();proxy.server_close();thread.join(timeout=2)
            # The source changed after its immutable manifest: fail, never spin at EOF.
            (cp/'training_state.pt').write_bytes(b'cut')
            eof_started=time.monotonic();eof_seen=False
            try:
                with urllib.request.urlopen(url+'/files/training_state.pt',timeout=3) as response:
                    response.read()
            except http.client.IncompleteRead as error:
                assert error.partial==b'cut';eof_seen=True
            assert eof_seen and time.monotonic()-eof_started<1 and proc.poll() is None
            checks['source_short_EOF_immediately_closes_with_IncompleteRead']=True
        finally:
            proc.terminate();proc.wait(timeout=5)
    result={'status':'PASS_LOCAL_CPU_FIXTURE_NOT_REMOTE_CHECKPOINT_EXPORT','checks':checks,'seconds':time.monotonic()-began,
        'source_sha256':{name:sha(HERE/name) for name in ['export_checkpoint_server.py','fetch_checkpoint.py','test_export_contract.py']},
        'no_MCP_no_GPU_no_model_files':True,'scope':'Synthetic bytes only; actual production checkpoint restore remains an explicit later check.'}
    with (HERE/'export-cpu-contract.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
