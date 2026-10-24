"""One original frozen Sap normalization after actual GP GPU release."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from durable import ref,sha,save,fsync_dir


def inventory(root,uuid):
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
    target=[[x.strip() for x in r.split(',')] for r in gpu.splitlines() if uuid in r]
    assert len(target)==1
    return {'observed_unix':time.time(),'all_gpus':gpu,'all_apps':apps,'target':target[0],
            'target_apps':[x for x in apps.splitlines() if uuid in x],'diskfree':shutil.disk_usage(root).free}


def main():
    root=Path(__file__).resolve().parent;cfg=json.loads((root/'norm-plan.json').read_text());ticket=json.loads((root/'launch-ticket.json').read_text())
    job=root/'job';job.mkdir(exist_ok=True)
    assert time.time()<=ticket['launch_not_after_unix'],'Expired normalization intent'
    with (job/'exclusive-launch.json').open('x') as f:json.dump(ticket,f);f.flush();os.fsync(f.fileno())
    fsync_dir(job);assert ticket['plan']==ref(root/'norm-plan.json')
    parent=Path(cfg['formal_root']);ended=json.loads((parent/'job/ended.json').read_text());complete=json.loads((parent/'output/complete.json').read_text())
    assert ended['status']=='COMPLETE' and ended['returncode']==0 and not ended['timed_out']
    assert complete['status']=='COMPLETE_FRESH8EP256_AND_ALL17_HELD_RAW' and complete['held_windows']==17 and complete['actual_adam_updates']==256
    assert sha(parent/'formal-plan.json')==cfg['formal_plan_sha256']
    assert ref(parent/'output/span-pool.jsonl')==complete['span_pool']
    for n,v in cfg['source_files'].items():assert ref(root/n)['sha256']==v['sha256']
    for k,v in cfg['upstream_files'].items():assert sha(v['path'])==v['sha256'],k
    for n,v in cfg['sap_assets'].items():assert sha(Path(cfg['sap_model'])/n)==v,n
    gate=inventory(root,cfg['gpu_uuid']);save(job/'fresh-gate.json',gate)
    assert gate['target'][0]=='0' and not gate['target_apps'] and int(gate['target'][3])<=32 and int(gate['target'][4])>=23000
    assert gate['diskfree']>=1_000_000_000
    save(root/'input-binding.json',{'formal_complete':ref(parent/'output/complete.json'),'formal_release':ref(parent/'job/ended.json'),
         'span_pool':complete['span_pool'],'upstream':cfg['upstream_files'],'old_public_Sap_only':True,'source':ref(root/'norm-plan.json')})
    assert time.time()<=ticket['launch_not_after_unix']
    cmd=[cfg['python'],'-u',str(root/'link_decode.py'),'--train',cfg['upstream_files']['train']['path'],'--input',cfg['upstream_files']['input']['path'],
         '--hpo',cfg['upstream_files']['hpo']['path'],'--span-pool',str(parent/'output/span-pool.jsonl'),'--sapbert',cfg['sap_model'],
         '--legacy-source',cfg['upstream_files']['legacy_source']['path'],'--semantic-source',cfg['upstream_files']['semantic_source']['path'],'--output',str(root/'linked')]
    start=time.time();deadline=start+180
    with (job/'stdout').open('wb') as stdout,(job/'stderr').open('wb') as stderr:
        child=subprocess.Popen(cmd,cwd=root,env={**os.environ,**cfg['environment'],'CUDA_VISIBLE_DEVICES':cfg['gpu_uuid']},stdout=stdout,stderr=stderr,start_new_session=True)
        save(job/'started.json',{'started_unix':start,'TERM_unix':deadline,'KILL_unix':deadline+30,'driver_pid':os.getpid(),'child_pid':child.pid,'plan':ref(root/'norm-plan.json')})
        timed=False
        try:rc=child.wait(timeout=max(.01,deadline-time.time()))
        except subprocess.TimeoutExpired:
            timed=True;os.killpg(child.pid,signal.SIGTERM)
            try:rc=child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);rc=child.wait()
    save(job/'ended.json',{'status':'COMPLETE' if rc==0 else 'FAILED','returncode':rc,'timed_out':timed,'seconds':time.time()-start,
         'started_unix':start,'ended_unix':time.time(),'release':inventory(root,cfg['gpu_uuid'])})
    sys.exit(rc)


if __name__=='__main__':main()
