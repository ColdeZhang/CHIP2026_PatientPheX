"""One hidden-CUDA CPU contract on the server holding the existing R1 cache."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    args=parser.parse_args();directory=Path(args.directory).resolve()
    deploy=json.loads((directory/'deployment.json').read_text())
    sys.path.insert(0,deploy['mcp_scripts']);from mcp_client import Client
    def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def save(name,value):
        with (directory/name).open('x') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    def unpack(value):return value.get('structuredContent') or json.loads(value['content'][0]['text'])
    save('execution-intent.json',dict(start_epoch=time.time(),deployment_sha256=sha(directory/'deployment.json'),
        scope='CPU59s+kill1 on existing cached R1; upload124knowntraining source only, noGPU/no model download/no quality',controller_sha256=sha(__file__)))
    client=Client();client.initialize()
    for name,item in deploy['upload_files'].items():
        path=Path(item['path']);assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes']
        binary=name.endswith('.gz')
        response=client.request('tools/call',{'name':'write_file','arguments':dict(mount_id=deploy['mount_id'],
            path=deploy['remote_relative']+'/'+name,content=base64.b64encode(path.read_bytes()).decode() if binary else path.read_text(),
            encoding='base64' if binary else 'utf8',create_parents=True)})
        save(name+'.upload.json',response);assert not response.get('isError')
    expected={n:v['sha256'] for n,v in deploy['upload_files'].items()}
    code='from pathlib import Path;import hashlib;p=Path('+repr(deploy['remote_absolute'])+');e='+repr(expected)+';assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in e.items());assert not (p/"actual").exists()'
    command='python3 -c '+shlex.quote(code)+' && timeout --signal=TERM --kill-after=1s 59s '+shlex.quote(deploy['python'])+' -u '+shlex.quote(deploy['remote_absolute']+'/cpu_contract.py')+' --plan '+shlex.quote(deploy['remote_absolute']+'/plan.json')
    save('command.json',dict(command=command,env=deploy['environment'],timeout_seconds=60))
    response=client.request('tools/call',{'name':'run_command','arguments':dict(mount_id=deploy['mount_id'],command=command,
        env=deploy['environment'],timeout_seconds=60)})
    save('actual-tool.json',response);value=unpack(response)
    print(json.dumps(dict(exit_code=value.get('exit_code'),stdout_tail=value.get('stdout','')[-1800:],stderr_tail=value.get('stderr','')[-1000:])))
    response=client.request('tools/call',{'name':'read_file','arguments':dict(mount_id=deploy['mount_id'],
        path=deploy['remote_relative']+'/actual/result.json',offset=0,limit=1000000)})
    save('result-read-tool.json',response);item=unpack(response);data=item['content'].encode()
    assert hashlib.sha256(data).hexdigest()==item['sha256']
    (directory/'actual-result.json').write_bytes(data)
    result=json.loads(data);recovered=directory/'recovered';recovered.mkdir(exist_ok=False)
    outputs=dict(result.get('token_contract',{}).get('outputs',{}))
    if result.get('parameter_contract',{}).get('actual_forward'):
        outputs['actual-qwen3-forward.txt']=result['parameter_contract']['actual_forward']
    for name,reference in outputs.items():
        if name.endswith('.gz'):continue
        response=client.request('tools/call',{'name':'read_file','arguments':dict(mount_id=deploy['mount_id'],
            path=deploy['remote_relative']+'/actual/'+name,offset=0,limit=1000000)})
        item=unpack(response);data=item['content'].encode()
        assert len(data)==reference['bytes'] and hashlib.sha256(data).hexdigest()==reference['sha256']==item['sha256']
        (recovered/name).write_bytes(data)
    save('collection.json',dict(status=result['status'],actual_exit_code=value.get('exit_code'),result_sha256=sha(directory/'actual-result.json'),
        program_seconds=result['program_seconds'],full_encoded_training_stays_in_owned_server_directory=True))
    assert value.get('exit_code')==0


if __name__=='__main__':main()
