"""One bounded tiny CPU contract on installed Torch; CUDA explicitly hidden."""
import argparse,base64,importlib.util,json,shlex
from pathlib import Path
from common import atomic,ref

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a=a.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);w=Path(__file__).parent
    spec=importlib.util.spec_from_file_location('mcp','/workspace/CHIP2026_PatientPheX/scripts/mcp_client.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);c=m.Client();c.initialize()
    mount='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/patient-set-risk-v1/cpu-loss-v1';remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+rel
    def call(name,args,label):
        v=c.request('tools/call',dict(name=name,arguments=args));atomic(out/(label+'.json'),v);assert not v.get('isError');return v.get('structuredContent') or json.loads(v['content'][0]['text'])
    for n in ('cpu_contract.py','common.py','scst_ops.py'):
        p=w/n;v=call('write_file',dict(mount_id=mount,path=rel+'/'+n,content=base64.b64encode(p.read_bytes()).decode(),encoding='base64',create_parents=True),'upload-'+n);assert v['sha256']==ref(p)['sha256']
    command='timeout --signal=TERM --kill-after=5 50 /home/zhangyuheng/projects/ScratchVL/.venv/bin/python '+shlex.quote(remote+'/cpu_contract.py')+' --output '+shlex.quote(remote+'/result.json')
    arguments=dict(mount_id=mount,command=command,cwd=rel,env={'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1'},timeout_seconds=55)
    atomic(out/'intent.json',dict(arguments=arguments,source=ref(__file__)));v=call('run_command',arguments,'response');assert v['exit_code']==0,v
    v=call('read_file',dict(mount_id=mount,path=rel+'/result.json',offset=0,limit=1048576),'readback')
    b=base64.b64decode(v['content']) if v.get('encoding')=='base64' else v['content'].encode();(out/'result.json').write_bytes(b)
    assert ref(out/'result.json')['sha256']==v['sha256'];atomic(out/'complete.json',dict(status='PASS_BOUND_TINY_CPU_ONLY',result=ref(out/'result.json'),source=ref(__file__),no_GPU=True))
    print(json.dumps(json.loads(b)))
if __name__=='__main__':main()
