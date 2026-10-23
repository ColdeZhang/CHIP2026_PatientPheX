#!/usr/bin/env python3
"""Single bounded own public-candidate JSON SFT job with fresh resource check and recovery record."""
from pathlib import Path
import argparse,json,hashlib,shlex,sys,time,datetime

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--client-directory',required=True);p.add_argument('--io-directory',required=True)
    a=p.parse_args();b=a.directory.resolve();plan=json.loads((b/'run-plan.json').read_text())
    assert sha(__file__)==plan['launcher_sha256'] and not (b/'job.json').exists()
    for ref in plan['references'].values():
        assert sha(ref['path'])==ref['sha256'] and Path(ref['path']).stat().st_size==ref['bytes']
    sys.path[:0]=[a.client_directory,a.io_directory];import remote_io as r;r.M=plan['mount_id']
    def save(name,value):
        with (b/name).open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    def readcall(name,args):
        for attempt in range(3):
            try:return r.call(name,args)
            except Exception:
                if attempt==2:raise
                time.sleep(2)
    def command(code):
        x=readcall('run_command',dict(command='python3 -c '+shlex.quote(code),cwd=plan['cwd'],timeout_seconds=60))
        assert x.get('exit_code')==0,x
        return x
    def gpu(name):
        code="import subprocess,json,time;u="+repr(plan['gpu_uuid'])+";a=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True);assert not any(x.split(',')[0].strip()==u for x in a.splitlines());g=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader'],text=True);d=next(x for x in g.splitlines() if x.split(',')[0].strip()==u);assert int(d.split(',')[2].strip().split()[0])>=70000;print(json.dumps(dict(uuid=u,empty=True,device=d,server_epoch=time.time(),apps=a)))"
        save(name,command(code))
    save('source-preflight.json',command("from pathlib import Path;import hashlib,shutil,json;expected="+repr(plan['remote_sources_sha256'])+";actual={k:hashlib.sha256(Path(k).read_bytes()).hexdigest() for k in expected};assert actual==expected;assert not Path("+repr(plan['remote_root']+'/run_manifest.json')+").exists();free=shutil.disk_usage('.').free;assert free>="+str(plan['minimum_free_disk_bytes'])+";print(json.dumps(dict(actual=actual,free_disk_bytes=free)))"))
    gpu('gpu-preflight.json');cap,grace=plan['cap_seconds'],plan['grace_seconds']
    assert (cap,grace) in {(2400,30),(600,30)}
    assert plan['argv'][:4]==['/usr/bin/timeout','--signal=TERM',f'--kill-after={grace}s',str(cap)]
    job=r.call('start_process',dict(command=shlex.join(plan['argv']),cwd=plan['cwd'],env=plan['env']))
    save('job.json',dict(job=job,plan_sha256=sha(b/'run-plan.json'),cap_seconds=cap,grace_seconds=grace,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(job=job['id'],cap=cap,grace=grace)),flush=True);cursors={'stdout':0,'stderr':0}
    while True:
        running=False
        for stream in cursors:
            x=readcall('read_process_output',dict(process_id=job['id'],stream=stream,cursor=cursors[stream],limit=20000))
            cursors[stream]=x['next_cursor'];running|=x['running']
            if x['content']:
                with (b/(stream+'.log')).open('a') as f:f.write(x['content'])
                if stream=='stdout':print(x['content'],flush=True)
        (b/'monitor.json').write_text(json.dumps(dict(job=job['id'],running=running,cursors=cursors,
            at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
        if not running:break
        time.sleep(10)
    gpu('gpu-release.json')
    collect=list(plan['collect'])
    for name in collect:
        remote=plan['remote_root']+'/'+name
        exists=command('from pathlib import Path;import json;print(json.dumps(Path('+repr(remote)+').is_file()))')
        if not json.loads(exists['stdout']):
            if name in plan.get('optional_collect',[]):continue
            save('incomplete.json',dict(missing=remote,no_automatic_retry=True));return
        x=readcall('read_file',dict(path=plan['cwd']+'/'+remote,offset=0,limit=1000000))
        assert hashlib.sha256(x['content'].encode()).hexdigest()==x['sha256']
        (b/name).parent.mkdir(parents=True,exist_ok=True);(b/name).write_text(x['content'])
    if plan.get('phase')=='train':
        terminal=json.loads((b/'training_terminal.json').read_text())
        cp=terminal.get('checkpoint',plan['remote_root']+'/epoch-002')
        if not cp.startswith(plan['remote_root']+'/'):raise RuntimeError('Checkpoint outside own run')
        for name in ['checkpoint.json']:
            x=readcall('read_file',dict(path=plan['cwd']+'/'+cp+'/'+name,offset=0,limit=1000000))
            assert hashlib.sha256(x['content'].encode()).hexdigest()==x['sha256']
            target=b/Path(cp).name/name;target.parent.mkdir(exist_ok=True);target.write_text(x['content'])
        report=command("from pathlib import Path;import hashlib,json;p=Path("+repr(cp)+");m=json.loads((p/'checkpoint.json').read_text());actual={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in m['files']};assert actual==m['files'];print(json.dumps({'path':str(p),'files':actual,'file_sizes':{n:(p/n).stat().st_size for n in m['files']},'complete_epochs':m['completed_epochs'],'global_step':m['global_step'],'partial_not_for_inference':m['partial_not_for_inference']}))")
        save('checkpoint-actual-hashes.json',json.loads(report['stdout']))
        if terminal['status']=='complete_epoch002_54':
            for name in ['training_complete.json','epoch-001-summary.json','epoch-002-summary.json']:
                x=readcall('read_file',dict(path=plan['cwd']+'/'+plan['remote_root']+'/'+name,offset=0,limit=1000000));assert hashlib.sha256(x['content'].encode()).hexdigest()==x['sha256'];(b/name).write_text(x['content'])
    print('COLLECTED_TERMINAL_GPU_RELEASED',flush=True)

if __name__=='__main__':main()
