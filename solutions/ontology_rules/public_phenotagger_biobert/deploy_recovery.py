"""Deploy independent CPU recovery, reusing only complete verified public weight."""
from pathlib import Path
import json,hashlib,base64,sys,shlex,concurrent.futures
R=Path(__file__).parent/'attempt2';sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
M='476f81cf-09e7-445b-a7f0-d0e12b12e50f';remote='public-phenotagger-biobert-attempt2';parent='public-phenotagger-biobert'
p=json.loads((R/'plan.json').read_text());names=[*p['source_sha256'],'plan.json']
def call(c,n,**a):
 x=c.request('tools/call',{'name':n,'arguments':{'mount_id':M,**a}});assert not x.get('isError');return x.get('structuredContent') or json.loads(x['content'][0]['text'])
def put(n):
 c=Client();c.initialize();return call(c,'write_file',path=remote+'/'+n,content=base64.b64encode((R/n).read_bytes()).decode(),encoding='base64',create_parents=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:list(e.map(put,names))
code="from pathlib import Path;import json,hashlib;src=Path('../"+parent+"');assert json.loads((src/'CPU-state.json').read_text())['status']=='failed_preserved';assert not Path('CPU-state.json').exists();[(Path(n).symlink_to(src/n)) for n in ['biobert_PT_v1.2.h5','weight-source-manifest.json']];names="+repr(names)+";print(json.dumps({n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in names}))"
c=Client();c.initialize();v=call(c,'run_command',command='/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code),cwd=remote,env={'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},timeout_seconds=30);assert v['exit_code']==0,v['stderr'];actual=json.loads(v['stdout']);assert all(h==hashlib.sha256((R/n).read_bytes()).hexdigest() for n,h in actual.items());(R/'deployment.json').write_text(json.dumps({'files':actual,'shared_readonly_weight':'../public-phenotagger-biobert/biobert_PT_v1.2.h5','no_weight_redownload':True},indent=2)+'\n');print('recovery deployment allSHA PASS')
