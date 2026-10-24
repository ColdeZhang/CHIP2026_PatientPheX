"""Single shared wall-clock budget: two warm-start arms, four exports, two fixed F1 inferences."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,math,os,shlex,sys,time

sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=lambda p:dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))


class Run:
    def __init__(self,directory):
        self.root=directory;self.plan=json.loads((directory/'run-plan.json').read_text())
        self.deadline=None
    def call(self,mount,name,args):
        c=Client();c.initialize();r=c.request('tools/call',{'name':name,'arguments':{'mount_id':mount,**args}})
        assert not r.get('isError'),'MCP tool error; inspect existing jobs before recovery'
        return r.get('structuredContent') or json.loads(r['content'][0]['text'])
    def save(self,path,obj):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    def command(self,mount,code,limit=30):
        if self.deadline is not None:limit=max(1,min(limit,math.floor(self.deadline-time.time())))
        r=self.call(mount,'run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','timeout_seconds':limit})
        assert r['exit_code']==0,r
        return r
    def remaining(self):
        n=math.floor(self.deadline-time.time())
        assert n>0,'Shared 3900-second deadline reached; preserve state, do not start another task'
        return n
    def gpu_empty(self,label):
        p=self.plan;code="import subprocess,json,shutil,time;u="+repr(p['gpu_uuid'])+";apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);assert not any(v.split(',')[0].strip()==u for v in apps.splitlines());s=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader'],text=True);d=next(v for v in s.splitlines() if v.split(',')[0].strip()==u);assert int(d.split(',')[2].strip().split()[0])>=76000;print(json.dumps(dict(gpu_uuid=u,empty=True,device=d,server_epoch=time.time(),shm_free=shutil.disk_usage('/dev/shm').free)))"
        r=self.command(p['mount_id'],code);self.save(self.root/(label+'.json'),r);return json.loads(r['stdout'])
    def watch(self,mount,job,directory):
        directory.mkdir(parents=True,exist_ok=True);cursors={'stdout':0,'stderr':0};started=time.monotonic()
        while True:
            running=False
            for stream in cursors:
                x=self.call(mount,'read_process_output',{'process_id':job['id'],'stream':stream,'cursor':cursors[stream],'limit':20000});cursors[stream]=x['next_cursor'];running|=x['running']
                if x['content']:
                    with (directory/(stream+'.log')).open('a') as f:f.write(x['content'])
                    if stream=='stdout':print(x['content'],flush=True)
            (directory/'monitor.json').write_text(json.dumps({'job':job['id'],'running':running,'cursors':cursors,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
            if not running:break
            # timeout is already enforced remotely; this monitor may continue through TERM grace.
            assert time.time()<self.deadline+90,'Remote state uncertain after deadline; inspect known job, never duplicate'
            time.sleep(8)
        self.save(directory/'process-terminal.json',{'job':job,'running':False,'remote_exit_code_available':False,'monitor_seconds':time.monotonic()-started,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    def start_bounded(self,mount,argv,cwd,env,path):
        self.remaining()
        # Recompute on the server: MCP queue/network time cannot extend the global cap.
        wrapper='import os,time;remaining=int('+repr(self.deadline)+'-time.time());assert remaining>0,\"Absolute campaign deadline reached before remote start\";os.execvp(\"/usr/bin/timeout\",[\"/usr/bin/timeout\",\"--signal=TERM\",\"--kill-after=30s\",str(remaining)]+'+repr(argv)+')'
        command=['python3','-c',wrapper]
        # Durable intent precedes mutating start; an ambiguous failure must be inspected.
        self.save(path.with_suffix('.intent.json'),{'argv':command,'payload_argv':argv,'mount_id':mount,'cwd':cwd,'env':env,
            'absolute_deadline_unix':self.deadline,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'intent_before_start_no_known_job_yet'})
        # Mutating start is never retried. Ambiguous network failure requires process inspection.
        job=self.call(mount,'start_process',{'command':shlex.join(command),'cwd':cwd,'env':env})
        self.save(path,{'job':job,'argv':command,'payload_argv':argv,'absolute_deadline_unix':self.deadline,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});return job
    def get_file(self,mount,remote,destination,record=None):
        if record is None:
            r=self.command(mount,'from pathlib import Path;import hashlib,json;p=Path('+repr(remote)+');assert p.is_file() and not p.is_symlink();print(json.dumps(dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))')
            record=json.loads(r['stdout'])
        data=bytearray()
        for offset in range(0,record['bytes'],512*1024):
            x=self.call(mount,'read_file',{'path':remote,'offset':offset,'limit':min(512*1024,record['bytes']-offset)})
            data.extend(base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode())
        assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert destination.read_bytes()==data
        else:destination.write_bytes(data)
        return json.loads(data) if destination.suffix=='.json' else None
    def shm_json(self,remote,destination):
        assert remote.startswith(self.plan['shm_root']+'/')
        x=self.command(self.plan['mount_id'],'from pathlib import Path;import base64,json;p=Path('+repr(remote)+');assert p.is_file() and p.stat().st_size<512*1024;print(json.dumps(dict(data=base64.b64encode(p.read_bytes()).decode())))')
        data=base64.b64decode(json.loads(x['stdout'])['data']);destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert destination.read_bytes()==data
        else:destination.write_bytes(data)
        return json.loads(data)
    def export(self,arm,epoch):
        p=self.plan;cp=p['shm_root']+'/'+arm+f'/epoch-{epoch:03d}';out=self.root/arm/f'export-{epoch:03d}';out.mkdir()
        job=self.start_bounded(p['mount_id'],['python3','-u',p['remote_root']+'/export_checkpoint_server.py','--checkpoint',cp],'.',{},out/'server-job.json')
        try:
            ready=None
            for _ in range(12):
                x=self.call(p['mount_id'],'read_process_output',{'process_id':job['id'],'stream':'stdout','cursor':0,'limit':30000})
                for line in x.get('content','').splitlines():
                    if line.startswith('{'):ready=json.loads(line)
                if ready:break
                assert x['running'],'Checkpoint server failed before ready';time.sleep(1)
            assert ready and ready['ready'];self.save(out/'server-ready.json',ready)
            destination=p['persistent_root']+'/'+arm+f'/epoch-{epoch:03d}'
            argv=['python3','-u',p['persistent_tools']+'/fetch_checkpoint.py','--url','http://10.5.4.7:'+str(ready['port']),
                '--manifest-sha256',ready['manifest_sha256'],'--destination',destination,'--deadline',str(self.deadline)]
            fetch=self.start_bounded(p['persistent_mount_id'],argv,'.',{},out/'fetch-job.json');self.watch(p['persistent_mount_id'],fetch,out/'fetch')
            code="from pathlib import Path;import hashlib,json;p=Path("+repr(destination)+");a=json.loads((p/'export-ack.json').read_text());assert a['status']=='persistent_checkpoint_verified';assert all((p/n).is_file() and not (p/n).is_symlink() and (p/n).stat().st_size==r['bytes'] and hashlib.sha256((p/n).read_bytes()).hexdigest()==r['sha256'] for n,r in a['files'].items());print(json.dumps(a))"
            actual=self.command(p['persistent_mount_id'],code,60);self.save(out/'persistent-actual-readback.json',actual);ack=json.loads(actual['stdout'])
            assert ack['files']==ready['manifest']['files'] and ack['checkpoint_sha256']==ready['manifest']['checkpoint_sha256'] and ack['source_checkpoint_directory']==cp
            self.save(out/'export-ack.json',ack)
            target=self.root/arm/f'epoch-{epoch:03d}'
            self.get_file(p['persistent_mount_id'],destination+'/checkpoint.json',target/'checkpoint.json',ack['files']['checkpoint.json'])
            self.get_file(p['persistent_mount_id'],destination+'/_run/run_manifest.json',target/'run_manifest.json',ack['files']['_run/run_manifest.json'])
            self.get_file(p['persistent_mount_id'],destination+'/_run/progress.json',target/'progress.json',ack['files']['_run/progress.json'])
            self.save(target/'export-ack.json',ack)
            receipt=cp.rsplit('/',1)[0]+f'/export-ack-epoch-{epoch:03d}.json';text=json.dumps(ack,sort_keys=True,indent=2)+'\n'
            check=self.command(p['mount_id'],'from pathlib import Path;import hashlib,json;p=Path('+repr(receipt)+');assert not p.exists();s='+repr(text)+';p.write_text(s);assert hashlib.sha256(p.read_bytes()).hexdigest()=='+repr(hashlib.sha256(text.encode()).hexdigest())+';print(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())))')
            self.save(out/'source-ack-written.json',check)
            return ack
        finally:
            self.save(out/'server-stopped.json',self.call(p['mount_id'],'stop_process',{'process_id':job['id']}))
    def train_segment(self,arm,epoch):
        p=self.plan;output=p['shm_root']+'/'+arm;path=self.root/arm/f'train-{epoch:03d}';path.mkdir(parents=True)
        self.gpu_empty(arm+f'/train-{epoch:03d}/gpu-preflight')
        argv=p['training_common_argv']+['--output-dir',output,'--order-arm',arm,'--stop-after-epoch',str(epoch)]
        if epoch==2:argv+=['--resume',output+'/epoch-001','--export-ack',output+'/export-ack-epoch-001.json']
        job=self.start_bounded(p['mount_id'],argv,'.',p['env'],path/'job.json');self.watch(p['mount_id'],job,path)
        self.gpu_empty(arm+f'/train-{epoch:03d}/gpu-release')
        progress=self.shm_json(output+'/progress.json',path/'progress.json');assert progress['completed_epochs']==epoch and progress['global_step']==16*epoch
        marker='paused.json' if epoch==1 else 'training_complete.json';result=self.shm_json(output+'/'+marker,path/marker)
        if epoch==1:assert result['completed_epochs']==1 and result['global_step']==16
        else:assert result['epochs']==2 and result['global_step']==32 and not result['smoke_only']
        return self.export(arm,epoch)
    def infer(self,arm,ack):
        p=self.plan;out=self.root/arm;cp=out/'epoch-002/checkpoint.json';complete=out/'train-002/training_complete.json'
        adapter=p['shm_root']+'/'+arm+'/epoch-002/adapter';adapter_files={n:ack['files']['adapter/'+n]['sha256'] for n in ['adapter_config.json','adapter_model.safetensors']}
        ip={'checkpoint_sha256':sha(cp),'training_complete_sha256':sha(complete),'adapter_files':adapter_files,'adapter_directory':adapter,
            'prompts_sha256':p['prompts_sha256'],'runner_sha256':p['runner_sha256'],'model_source_sha256':p['model_source_sha256'],'rows':80,'no_F1_gold_opened':True}
        self.save(out/'inference-plan.json',ip);self.gpu_empty(arm+'/inference-gpu-preflight')
        remote=p['remote_root']+'/inference/'+arm+'/responses.jsonl'
        argv=p['inference_common_argv']+['--output',remote,'--adapter',adapter]
        job=self.start_bounded(p['mount_id'],argv,'.',p['env'],out/'inference-job.json');self.watch(p['mount_id'],job,out/'inference')
        self.gpu_empty(arm+'/inference-gpu-release')
        for suffix in ['jsonl','manifest.json','checkpoint.json','loading.json']:self.get_file(p['mount_id'],remote[:-5]+suffix,out/('responses.'+suffix))
        rows=[json.loads(x) for x in (out/'responses.jsonl').read_text().splitlines()];meta=json.loads((out/'responses.manifest.json').read_text());fp=meta['fingerprint']
        assert len(rows)==len({x['row_id'] for x in rows})==meta['expected_rows']==80
        assert {x['row_id']:str(x['pmc_id']) for x in rows}==p['row_documents']
        assert meta['output_sha256']==sha(out/'responses.jsonl') and fp['adapter_files']==adapter_files and fp['source_sha256']==p['runner_sha256'] and fp['prompts_sha256']==p['prompts_sha256']
        self.save(out/'responses-freeze.json',{'plan_sha256':sha(out/'inference-plan.json'),'raw_sha256':sha(out/'responses.jsonl'),'manifest_sha256':sha(out/'responses.manifest.json'),'rows':80,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    def execute(self):
        p=self.plan;assert sha(__file__)==p['controller_sha256']
        for r in p['references'].values():assert ref(r['path'])==r
        assert not (self.root/'execution-state.json').exists(),'Recover recorded existing jobs, never restart this campaign'
        assert p['cap_seconds']==3900 and p['grace_seconds']==30
        self.gpu_empty('initial-gpu-preflight')
        check=self.command(p['mount_id'],'from pathlib import Path;import hashlib,json,os,shutil;expected='+repr(p['remote_source_sha256'])+';assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in expected.items());root=Path('+repr(p['shm_root'])+');assert not root.exists();assert shutil.disk_usage("/dev/shm").free>4*1024**3;assert shutil.disk_usage(".").free>20*1024**2;root.mkdir(mode=0o700);assert root.stat().st_uid==os.getuid()==1045;print(json.dumps(dict(shm=str(root),free=shutil.disk_usage("/dev/shm").free)))')
        self.save(self.root/'initial-source-and-storage.json',check)
        check=self.command(p['persistent_mount_id'],'from pathlib import Path;import hashlib,shutil,json;p=Path('+repr(p['persistent_root'])+');assert not p.exists();assert shutil.disk_usage(".").free>4*1024**3;assert hashlib.sha256(Path('+repr(p['persistent_tools']+'/fetch_checkpoint.py')+').read_bytes()).hexdigest()=='+repr(p['fetcher_sha256'])+';print(json.dumps(dict(persistent_free=shutil.disk_usage(".").free)))')
        self.save(self.root/'persistent-storage-preflight.json',check)
        started=time.time();self.deadline=started+3900
        state={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'started_unix':started,'absolute_deadline_unix':self.deadline,'TERM_latest_unix':self.deadline+30,'plan_sha256':sha(self.root/'run-plan.json'),'status':'running'}
        self.save(self.root/'execution-state.json',state)
        for arm in ['original','reverse-first']:
            for epoch in [1,2]:ack=self.train_segment(arm,epoch)
            self.infer(arm,ack)
        self.save(self.root/'all-responses-complete.json',{'status':'both_full_newstage_arms_and_durable_checkpoints_complete_before_quality','seconds':time.time()-started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_deadline_unix':self.deadline,'no_quality_evaluation':True})
        print('BOTH_ARMS_COMPLETE_AND_DURABLE_NO_QUALITY_YET',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path);a=p.parse_args();Run(a.directory.resolve()).execute()


if __name__=='__main__':main()
