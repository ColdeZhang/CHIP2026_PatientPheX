"""Prepare or execute exactly one already-authorized NCR CPU stage, with no polling loop."""
from pathlib import Path
import argparse, base64, hashlib, json, shlex, sys

R = Path(__file__).resolve().parent
BASE = R.parent / 'local-relay-attempt3'
PROJECT = '/home/zhangyuheng/projects/CHIP2026_PatientPheX'
DEST = 'public-neuralcr-readiness/postdownload-readiness-v1'
REMOTE = PROJECT+'/'+DEST
MOUNT = '476f81cf-09e7-445b-a7f0-d0e12b12e50f'
PYTHON = PROJECT+'/public-phenotagger-reuse/.venv/bin/python'
OVERLAY = PROJECT+'/public-neuralcr-readiness/runtime'
STAGES = {'shape': ('shape-plan.json', 'shape-inventory.json'),
          'repair': ('repair-plan.json', 'repair-report.json'),
          'native': ('native-cpu-plan.json', 'native-cpu-result.json')}
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(p.read_bytes())
write = lambda p, v: p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')
ref = lambda p: {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
parser = argparse.ArgumentParser()
parser.add_argument('action', choices=['prepare','start','collect'])
parser.add_argument('stage', choices=STAGES)
a = parser.parse_args()
planfile, resultfile = STAGES[a.stage]
state = read(BASE/'state.json')
assert state['status'] == 'complete_two_official_assets_verified_native_not_run'
hp, ft = state['assets']
assert hp['filename'] == 'HP.tar.gz' and ft['filename'] == 'pmc_model_new.bin'

if a.action == 'prepare':
    assert not (R/planfile).exists()
    if a.stage == 'shape':
        (R/'HP.tar.gz.assembly.json').write_bytes((BASE/'HP.tar.gz.assembly.json').read_bytes())
        inputs = {'fastText': {k:ft[k] for k in ['path','bytes','sha256']},
                  'HP_report': {'path': REMOTE+'/HP.tar.gz.assembly.json', **{k:v for k,v in ref(R/'HP.tar.gz.assembly.json').items() if k!='path'}}}
        for i, value in enumerate(hp['extracted']): inputs['HP_'+str(i)] = value
        plan = {'scope':'complete official assets inventory only; no model/NN or competition data',
                'inputs': inputs, 'HP_report': REMOTE+'/HP.tar.gz.assembly.json',
                'fastText_file': ft['path'], 'relay_plan_sha256': state['plan_sha256']}
    elif a.stage == 'repair':
        inventory = read(R/'shape-inventory.json')
        assert inventory['status']=='complete_inventory' and not inventory['fastText']['standard_layout_valid']
        assert inventory['fastText']['EOF_exact']
        plan = {'original_file':ft['path'], 'original_bytes':ft['bytes'], 'original_sha256':ft['sha256'],
                'inventory_path':REMOTE+'/shape-inventory.json','inventory_sha256':sha((R/'shape-inventory.json').read_bytes()),
                'policy':'Only declared 12-byte serialization correction, original and all numerical parameter bytes unchanged',
                'preparation_sha256':sha((R/'header-repair-preparation.json').read_bytes())}
    else:
        inventory=read(R/'shape-inventory.json')
        if inventory['fastText']['standard_layout_valid']:
            fasttext_file='pmc_model_new.bin'; modelref=ft
        else:
            repair=read(R/'repair-report.json')
            assert repair['status']=='pass_exact_12byte_layout_repair' and repair['all_numerical_parameter_bytes_unchanged']
            fasttext_file='pmc_model_new.standard-layout.bin'; modelref=repair['derived']
        required={fasttext_file:{k:modelref[k] for k in ['bytes','sha256']}}
        for x in hp['extracted']:
            required['extracted/HP/'+Path(x['path']).name]={k:x[k] for k in ['bytes','sha256']}
        for n in ['shape-inventory.json','ncrmodel.py','check_native.py']:
            required[n]={k:v for k,v in ref(R/n).items() if k!='path'}
        plan={'runtime_overlay':OVERLAY,'fasttext_load_file':fasttext_file,'required_files':required,
              'sources':{'check_native.py':ref(R/'check_native.py')},
              'fixed_synthetic_texts':read(R/'preparation.json')['fixed_synthetic_texts'],
              'scope':'Only four synthetic CPU interface cases; no task inputs, quality or GPU'}
    write(R/planfile,plan)
    print(json.dumps({'prepared':planfile,'sha256':sha((R/planfile).read_bytes())}))
    raise SystemExit()

sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
client=Client();client.initialize()
def call(name, args):
    result=client.request('tools/call', {'name':name,'arguments':{'mount_id':MOUNT,**args}})
    assert not result.get('isError'), result
    return result['structuredContent']
def fetch(name):
    result=call('read_file',{'path':DEST+'/'+name,'offset':0,'limit':524288})
    b=base64.b64decode(result['content']) if result.get('encoding')=='base64' else result['content'].encode()
    (R/name).write_bytes(b)
    return b
if a.action == 'collect':
    job=read(R/(a.stage+'-managed-job.json'))
    result=call('read_process_output',{'process_id':job['id'],'stream':'stderr','cursor':0,'limit':4096})
    write(R/(a.stage+'-process-status.json'),result)
    if result['running']:
        print(json.dumps({'stage':a.stage,'running':True}));raise SystemExit()
    terminal=json.loads(fetch(a.stage+'-job-state.json'))
    fetch(a.stage+'.stderr.log');fetch(a.stage+'.stdout.log')
    if terminal['returncode']==0:fetch(resultfile)
    else:
        try:fetch(resultfile)
        except Exception:pass
    print(json.dumps({'stage':a.stage,**terminal}));raise SystemExit()

assert (R/planfile).is_file() and not (R/(a.stage+'-managed-job.json')).exists()
files=['stage_runner.py','inspect_complete.py','repair_fasttext_header.py','check_native.py','ncrmodel.py',planfile]
files += [n for n in ['HP.tar.gz.assembly.json','shape-inventory.json','repair-report.json'] if (R/n).exists()]
for name in files:
    b=(R/name).read_bytes();assert len(b)<=524288
    response=call('write_file',{'path':DEST+'/'+name,'content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True})
    assert response['sha256']==sha(b) and response['size']==len(b)
links={REMOTE+'/extracted/HP/'+Path(x['path']).name:x['path'] for x in hp['extracted']}
links[REMOTE+'/pmc_model_new.bin']=ft['path']
digests={name:sha((R/name).read_bytes()) for name in files}
code="from pathlib import Path;import os,hashlib,json\nlinks="+repr(links)+"\nfor dst,src in links.items():\n p=Path(dst);p.parent.mkdir(parents=True,exist_ok=True)\n if not p.exists():os.link(src,dst)\n assert os.path.samefile(src,dst)\nexpected="+repr(digests)+"\nactual={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in expected};assert actual==expected;print(json.dumps(actual))"
response=call('run_command',{'command':shlex.quote(PYTHON)+' -c '+shlex.quote(code),'cwd':DEST,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':30})
assert response['exit_code']==0,response
write(R/(a.stage+'-deployment.json'),{'files':digests,'remote_actual':json.loads(response['stdout']),'asset_hardlinks':links})
env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TF_NUM_INTRAOP_THREADS':'1','TF_NUM_INTEROP_THREADS':'1'}
command=shlex.quote(PYTHON)+' stage_runner.py '+a.stage
job=call('start_process',{'command':command,'cwd':DEST,'env':env})
write(R/(a.stage+'-managed-job.json'),job)
write(R/(a.stage+'-launch.json'),{'command':command,'env':env,'cwd':DEST,'plan_sha256':sha((R/planfile).read_bytes()),'launcher_sha256':sha(Path(__file__).read_bytes())})
print(json.dumps(job))
