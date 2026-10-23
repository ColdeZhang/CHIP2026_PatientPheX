#!/usr/bin/env python3
"""Run one frozen GLiNER augmentation control; no gold reads or automatic retries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    out = Path(plan['state_dir']);out.mkdir(parents=True,exist_ok=True)
    state_path = out/'state.json'
    assert not state_path.exists(), 'Never silently rerun or overwrite a control'
    assert plan['training_steps'] == 3760 and plan['training_epochs'] == 10
    assert plan['training_timeout_seconds'] == 1500
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == plan['gpu_uuid']
    for path,digest in plan['source_and_input_sha256'].items():
        assert sha(path) == digest,path
    apps = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                           '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout
    assert not any(line.strip().startswith(plan['gpu_uuid']+',') for line in apps.splitlines()), 'Assigned GPU is occupied'
    gpu = subprocess.run(['nvidia-smi','--id='+plan['gpu_uuid'],
                          '--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],
                         check=True,capture_output=True,text=True).stdout.strip()
    assert int(gpu.rsplit(',',1)[1].strip()) > 8000
    state = {'plan_sha256':sha(args.plan),'source_sha256':sha(__file__),
             'started_at_utc':now(),'gpu_preflight':gpu,'compute_apps_preflight':apps,
             'scope':'one frozen training and input-only inference; no heldout gold',
             'stages':[],'complete':False}
    save(state_path,state)
    for stage in plan['stages']:
        stage_out = Path(stage['output'])
        assert not stage_out.exists(),stage_out
        expected_timeout = 1500 if stage['name']=='train' else 300
        assert stage['timeout_seconds'] == expected_timeout
        event={'name':stage['name'],'command':stage['command'],'started_at_utc':now(),
               'timeout_seconds':stage['timeout_seconds']}
        state['stages'].append(event);save(state_path,state)
        print(json.dumps({'stage':stage['name'],'started_at_utc':event['started_at_utc']}),flush=True)
        start=time.monotonic()
        with (out/(stage['name']+'.stdout.log')).open('w') as stdout, (out/(stage['name']+'.stderr.log')).open('w') as stderr:
            process=subprocess.Popen(stage['command'],stdout=stdout,stderr=stderr,start_new_session=True)
            event['pid']=process.pid;save(state_path,state)
            try:
                return_code=process.wait(timeout=stage['timeout_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
                event.update(timeout=True,return_code=process.returncode,finished_at_utc=now(),
                             seconds=time.monotonic()-start)
                state['failure']='stage hard timeout';save(state_path,state)
                raise RuntimeError('Frozen stage exceeded hard timeout') from None
        event.update(return_code=return_code,finished_at_utc=now(),seconds=time.monotonic()-start)
        save(state_path,state)
        assert return_code==0,stage['name']+' failed; no automatic retry'
        if stage['name']=='train':
            report=json.loads((stage_out/'run_manifest.json').read_text())
            assert report['completed_epochs']==10 and report['total_steps']==3760 and report['rows']==1503
            event['run_manifest_sha256']=sha(stage_out/'run_manifest.json')
            event['final_model_sha256']=sha(stage_out/'epoch-010/pytorch_model.bin')
        else:
            manifest=json.loads((stage_out/'manifest.json').read_text())
            assert manifest['input_sha256']==plan['heldout_input_sha256']
            frozen={'frozen_at_utc':now(),'gold_read':False,
                    'candidate_sha256':sha(stage_out/'candidates.jsonl'),
                    'raw_span_sha256':sha(stage_out/'raw-spans.jsonl'),
                    'manifest_sha256':sha(stage_out/'manifest.json'),
                    'input_sha256':manifest['input_sha256'],'stage':stage['name'],
                    'model_sha256':next(v for k,v in manifest['files'].items() if k.endswith('/pytorch_model.bin')),
                    'source_sha256':manifest['source_sha256']}
            save(stage_out/'prediction-freeze.json',frozen);event['prediction_freeze']=frozen
        save(state_path,state)
        print(json.dumps({'stage':stage['name'],'completed_at_utc':now(),'seconds':event['seconds']}),flush=True)
    state.update(complete=True,completed_at_utc=now());save(state_path,state)
    print(json.dumps({'complete':True,'completed_at_utc':state['completed_at_utc']}),flush=True)


if __name__=='__main__':
    main()
