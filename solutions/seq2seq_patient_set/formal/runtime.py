"""Explicit host binding, actual inputs and one free GPU; no implicit launch."""
import gzip
import json
import os
from pathlib import Path
import subprocess
import time
from common import ref, save


def same(path, expected):
    actual=ref(path);assert all(actual[k]==expected[k] for k in ('bytes','sha256')),(str(path),actual,expected)
    return actual


def bind(plan_path, binding_path, phase):
    p=json.loads(Path(plan_path).read_text());b=json.loads(Path(binding_path).read_text())
    assert p['schema']=='longt5-patient-set-formal-v1' and b['phase']==phase
    assert b['authorized_by_root'] is True and time.time()<b['admit_until']
    same(plan_path,b['plan']);app=Path(__file__).resolve().parent.parent
    for relative,expected in p['sources'].items():same(app/relative,expected)
    out=Path(b['output']);out.mkdir(parents=True,exist_ok=False)
    save(out/'intent.json',{'plan':ref(plan_path),'binding':ref(binding_path),'issued':time.time(),'phase':phase})
    props=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    selected=[x.split(',') for x in props.splitlines() if x.split(',')[0].strip()==b['gpu_uuid']]
    save(out/'admission.json',{'gpus':props,'apps':apps,'expected_uuid':b['gpu_uuid']})
    assert len(selected)==1 and int(selected[0][1])>=20000 and b['gpu_uuid'] not in apps
    os.environ['CUDA_VISIBLE_DEVICES']=b['gpu_uuid']
    modeldir=Path(b['model_directory'])
    for name,item in p['model'].items():same(modeldir/name,item)
    manifest=Path(b['supply_directory'])/'manifest.json';same(manifest,p['supply']);s=json.loads(manifest.read_text())
    key='train' if phase=='train' else 'held_input';data=manifest.parent/Path(s['artifacts'][key]['path']).name
    same(data,s['artifacts'][key]);rows=[json.loads(x) for x in gzip.decompress(data.read_bytes()).splitlines()]
    assert len(rows)==(87 if phase=='train' else 39) and len({r['row_id'] for r in rows})==len(rows)
    assert len({str(r['pmc_id']) for r in rows})==(32 if phase=='train' else 16)
    assert all(0<len(r['input_ids'])<=16384 and r['input_ids'][-1]==1 for r in rows)
    if phase=='infer':assert all('labels' not in r and 'target_text' not in r for r in rows)
    assert time.time()<b['admit_until'],'Expired before model load'
    return p,b,out,modeldir,rows
