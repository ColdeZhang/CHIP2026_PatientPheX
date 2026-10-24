"""Deploy only new private sources and execute one CUDA-hidden meta/token contract."""
from pathlib import Path
import datetime,hashlib,json,shlex,sys,time

R=Path(__file__).parent
BASE='case-retrieval/qwen35-preparation'
REMOTE=BASE+'/order-warmstart-stage'
sys.path[:0]=['/workspace/CHIP2026_PatientPheX/scripts','/workspace/patientphex-runs/round2-20261023/case_retrieval/setup']
import remote_io as io
io.M='69b991ee-183f-4add-af1a-579af92479c9'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name,obj):
    with (R/name).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')


def main():
    assert not (R/'cpu-runtime-job.json').exists(),'Existing CPU job must be recovered, never duplicated'
    check=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path; assert not Path('+repr(REMOTE)+').exists()'),'cwd':'.','timeout_seconds':10})
    assert check['exit_code']==0
    files={name:R/name for name in ['train_order_warmstart.py','qwen35_smoke.py','qwen35_relation_cost.py','export_checkpoint_server.py']}
    files['reverse124.jsonl']=R.parent/'candidate-order-training-feasibility/reverse124.jsonl'
    expected={REMOTE+'/'+n:sha(p) for n,p in files.items()}
    save('deployment-plan.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'remote_sha256':expected,'scope':__doc__,'training_not_started':True})
    for n,p in files.items():io.upload(str(p),REMOTE+'/'+n)
    check=io.call('run_command',{'command':'python3 -c '+shlex.quote('from pathlib import Path;import hashlib,json;expected='+repr(expected)+';actual={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert actual==expected;print(json.dumps(actual))'),'cwd':'.','timeout_seconds':30})
    assert check['exit_code']==0;save('deployment-actual.json',check)
    argv=['timeout','--signal=TERM','--kill-after=15s','120s','/home/zhangyuheng/projects/CHIP2026_PatientPheX/ontology-round2/qwen-venv/bin/python','-u',REMOTE+'/train_order_warmstart.py','--mode','cpu',
        '--train',BASE+'/sft-meta1-fusion/train.jsonl','--excluded-input',BASE+'/sft-meta1-fusion/excluded.input.jsonl','--data-manifest',BASE+'/sft-meta1-fusion/preparation-manifest.json',
        '--model','/dev/shm/patientphex-qwen35-1045-c20223623576','--model-source-manifest',BASE+'/model-source-manifest.json','--output-dir',REMOTE+'/cpu-original',
        '--warm-start-checkpoint',BASE+'/sft-meta1-fusion/formal2ep/epoch-002','--reverse-train',REMOTE+'/reverse124.jsonl','--order-arm','original']
    env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+BASE+'/a800-migration/runtime','OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    job=io.call('start_process',{'command':shlex.join(argv),'cwd':'.','env':env})
    save('cpu-runtime-job.json',{'job':job,'argv':argv,'env':env,'GPU':False,'cap_seconds':120,'grace_seconds':15})
    cursors={'stdout':0,'stderr':0}
    while True:
        running=False
        for stream in cursors:
            x=io.call('read_process_output',{'process_id':job['id'],'stream':stream,'cursor':cursors[stream],'limit':20000});cursors[stream]=x['next_cursor'];running|=x['running']
            if x['content']:
                with (R/('cpu-runtime-'+stream+'.log')).open('a') as f:f.write(x['content'])
                if stream=='stdout':print(x['content'],flush=True)
        (R/'cpu-runtime-monitor.json').write_text(json.dumps({'running':running,'cursors':cursors,'job':job['id'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
        if not running:break
        time.sleep(3)
    code='from pathlib import Path;import json,hashlib;p=Path('+repr(REMOTE+'/cpu-original')+');print(json.dumps({str(q.relative_to(p)):{"bytes":q.stat().st_size,"sha256":hashlib.sha256(q.read_bytes()).hexdigest()} for q in p.glob("*.json")}))'
    listing=io.call('run_command',{'command':'python3 -c '+shlex.quote(code),'cwd':'.','timeout_seconds':30});assert listing['exit_code']==0
    files=json.loads(listing['stdout']);assert 'cpu_complete.json' in files,'CPU contract did not complete; preserve failure'
    out=R/'actual-cpu';out.mkdir(exist_ok=False)
    for name,record in files.items():
        assert Path(name).name==name and record['bytes']<512*1024
        x=io.call('read_file',{'path':REMOTE+'/cpu-original/'+name,'offset':0,'limit':512*1024});data=x['content'].encode();assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'];(out/name).write_bytes(data)
    assert json.loads((out/'cpu_complete.json').read_text())['cuda_initialized'] is False
    save('cpu-runtime-complete.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'only_CPU_contract':True,'no_training_or_quality':True})
    print('ACTUAL_CPU_COMPLETE_NO_CUDA',flush=True)


if __name__=='__main__':main()
