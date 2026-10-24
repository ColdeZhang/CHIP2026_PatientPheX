"""Bound transport and a single admitted start; UNKNOWN never auto-retries."""
import argparse,base64,hashlib,importlib.util,json,os,shlex,time
from pathlib import Path
from common import atomic,exclusive,ref,sha
P=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')

class Remote:
    def __init__(self,mount,python,output):
        self.mount=mount;self.python=python;self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.counter=0
        s=importlib.util.spec_from_file_location('mcp','/workspace/CHIP2026_PatientPheX/scripts/mcp_client.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);self.client=m.Client();self.client.initialize()
    def call(self,name,args,label):
        self.counter+=1;stem=f'{self.counter:03d}-{label}';exclusive(self.output/(stem+'-intent.json'),dict(tool=name,arguments={k:v for k,v in args.items() if k!='content'}))
        x=self.client.request('tools/call',dict(name=name,arguments=dict(mount_id=self.mount,**args)));exclusive(self.output/(stem+'-response.json'),x)
        assert not x.get('isError'),('MCP outcome saved; recover original intent',stem)
        return x.get('structuredContent') or json.loads(x['content'][0]['text'])
    def code(self,code,cwd,label,env=None,timeout=30):
        x=self.call('run_command',dict(command=shlex.quote(self.python)+' -c '+shlex.quote(code),cwd=str(Path(cwd).relative_to(P)),env=env or {'CUDA_VISIBLE_DEVICES':''},timeout_seconds=timeout),label)
        assert x['exit_code']==0,x;return json.loads(x['stdout'])
    def put(self,local,remote,label):
        p=Path(local);x=self.call('write_file',dict(path=str(Path(remote).relative_to(P)),content=base64.b64encode(p.read_bytes()).decode(),encoding='base64',create_parents=True),label)
        assert x['sha256']==sha(p);return x
    def get(self,remote,local,label):
        target=Path(local);target.parent.mkdir(parents=True,exist_ok=True);offset=0;identity=None
        with target.open('xb') as f:
            while True:
                x=self.call('read_file',dict(path=str(Path(remote).relative_to(P)),offset=offset,limit=1048576),label+'-'+str(offset))
                identity=identity or (x['size'],x['sha256']);assert identity==(x['size'],x['sha256'])
                b=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();assert b or offset==identity[0]
                f.write(b);offset+=len(b)
                if offset==identity[0]:break
                assert offset<identity[0]
            f.flush();os.fsync(f.fileno())
        assert ref(target)['sha256']==identity[1];return ref(target)

def deploy(api,members,root):
    expected={}
    for n,x in members.items():
        p=Path(x['path']);assert ref(p)==x;api.put(p,root/n,'upload-'+n.replace('/','_'));expected[n]={k:x[k] for k in ('bytes','sha256')}
    code="import pathlib,json,hashlib,os; ex="+repr(expected)+"; actual={};\nfor n,w in ex.items():\n p=pathlib.Path(n);b=p.read_bytes();a={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};assert a==w,n\n with p.open('rb') as f:os.fsync(f.fileno())\n actual[n]=a\nfor p in sorted({pathlib.Path('.')}|{pathlib.Path(n).parent for n in ex},key=lambda x:len(x.parts),reverse=True):\n fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)\nprint(json.dumps({'status':'PASS_FSYNC_REOPEN_DEPLOYED','files':actual}))"
    return api.code(code,root,'deployed-fsync-reopen')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['deploy','start','status','fetch8','deliver']);ap.add_argument('--binding',required=True);ap.add_argument('--output',required=True);ap.add_argument('--release');ap.add_argument('--rewards');a=ap.parse_args()
    bpath=Path(a.binding).resolve();b=json.loads(bpath.read_text());e=json.loads(Path(b['execution']['path']).read_text());assert sha(b['execution']['path'])==b['execution']['sha256']
    assert sha(__file__)==e['orchestration_source']['sha256']
    root=Path(b['remote_root']);api=Remote(b['mount'],b['python'],a.output);out=Path(a.output)
    if a.mode=='deploy':
        result=deploy(api,b['members'],root);atomic(out/'complete.json',dict(result,binding=ref(bpath),source=ref(__file__)));return
    if a.mode=='status':
        code="import pathlib,json,subprocess;r={};\nfor n in ('launch.accepted.json','driver-start.json','child.json','driver-complete.json','terminal/progress.json','terminal/reward-request.json','terminal/loss-progress.json','terminal/result.json'):\n p=pathlib.Path(n)\n if p.exists():\n  x=json.loads(p.read_text());r[n]=x if n not in ('terminal/progress.json','terminal/loss-progress.json','terminal/result.json') else {k:x.get(k) for k in ('status','phase','completed','rows','error','program_seconds','supply_gate','optimizer_steps','real_backward_calls')}\nr['gpu']=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout;r['apps']=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout\nprint(json.dumps(r))"
        value=api.code(code,root,'status');atomic(out/'status.json',value);print(json.dumps(value));return
    if a.mode=='fetch8':
        api.get(root/'terminal/reward-request.json',out/'reward-request.json','request');request=json.loads((out/'reward-request.json').read_text());assert request['execution']['sha256']==b['execution']['sha256']
        got=api.get(root/'terminal/responses.json',out/'responses.json','raw8');assert got['sha256']==request['responses']['sha256']
        atomic(out/'complete.json',dict(request=ref(out/'reward-request.json'),responses=got,binding=ref(bpath)));return
    if a.mode=='deliver':
        assert a.rewards;reward=Path(a.rewards).resolve();x=json.loads(reward.read_text());assert len(x['rows'])==8 and len(x['pairs'])==4
        original=x['original_raw_responses'];delivery=dict(responses_sha256=original['sha256'],rewards_sha256=sha(reward),binding=ref(bpath),normalizer_manifest_sha256=e['normalizer_manifest_sha256'])
        atomic(out/'reward-delivery.json',delivery)
        # Publish delivery first, reward last via atomic rename: the waiting model cannot observe a half file.
        api.put(out/'reward-delivery.json',root/'reward-delivery.pending','delivery');api.put(reward,root/'rewards.pending','rewards')
        code="import pathlib,os,json,hashlib;ex="+repr({'reward-delivery.pending':sha(out/'reward-delivery.json'),'rewards.pending':sha(reward)})+";\nfor n,h in ex.items():\n p=pathlib.Path(n);assert hashlib.sha256(p.read_bytes()).hexdigest()==h\n with p.open('rb') as f:os.fsync(f.fileno())\nfor old,new in [('reward-delivery.pending','reward-delivery.json'),('rewards.pending','rewards.json')]:\n assert not pathlib.Path(new).exists();os.rename(old,new)\nfd=os.open('.',os.O_RDONLY);os.fsync(fd);os.close(fd)\nprint(json.dumps({'status':'REWARDS_ATOMICALLY_DELIVERED'}))"
        atomic(out/'complete.json',api.code(code,root,'publish-rewards'));return
    assert a.release,'Actual assigned main GPU full-chain release receipt required'
    release=json.loads(Path(a.release).read_text());assert release['whole_chain_released'] is True and release['gpu_uuid']==e['gpu_uuid']
    for n,x in json.loads((Path(__file__).parent/'normalization-source-manifest.json').read_text()).items():assert sha(Path(__file__).parent/n)==x['sha256']
    assert sha(Path(__file__).parent/'normalization-source-manifest.json')==e['normalizer_manifest_sha256']
    code="import pathlib,json,hashlib,subprocess,shutil;assert hashlib.sha256(pathlib.Path('execution.json').read_bytes()).hexdigest()=="+repr(b['execution']['sha256'])+";r={'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free','--format=csv,noheader,nounits'],text=True),'apps':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True),'disk':shutil.disk_usage('.').free};print(json.dumps(r))"
    fresh=api.code(code,root,'fresh-resource');g={l.split(',')[0].strip():[int(t) for t in l.split(',')[1:]] for l in fresh['gpu'].splitlines()}
    assert g[e['gpu_uuid']][0]<=64 and g[e['gpu_uuid']][1]>=38000 and fresh['disk']>=1024**3 and e['gpu_uuid'] not in fresh['apps']
    t=time.time();ad=dict(valid_from_unix=t,valid_until_unix=t+300,execution_sha256=b['execution']['sha256'],actual_prior_owner_release=ref(a.release))
    exclusive(bpath.parent/'unique-start-intent.json',dict(admission=ad,source=ref(__file__),binding=ref(bpath)));atomic(out/'admission.json',ad)
    api.put(out/'admission.json',root/'admission.json','admission');assert time.time()<ad['valid_until_unix']
    cmd=shlex.join([b['python'],'-u',str(root/'pilot_driver.py'),'--execution',str(root/'execution.json'),'--admission',str(root/'admission.json')])
    response=api.call('start_process',dict(command=cmd,cwd=str(root.relative_to(P)),env=b['environment']),'single-start');print(json.dumps(response))
if __name__=='__main__':main()
