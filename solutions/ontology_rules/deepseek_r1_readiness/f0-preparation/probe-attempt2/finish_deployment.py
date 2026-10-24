"""Fix only deployment command placeholder; retain original SyntaxError, no reupload or GPU."""
from pathlib import Path
import ast,hashlib,json,sys,shlex,os,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
P=Path(__file__).parent;M='69b991ee-183f-4add-af1a-579af92479c9';remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/f0-longest-v2';expected=json.loads((P/'plan.json').read_text())['sources'];digest=hashlib.sha256((P/'sources.tar.gz').read_bytes()).hexdigest()
s=(P/'deploy_once.py').read_text();s=s.replace(".replace('SHA',repr(digest))",".replace('==SHA','=='+repr(digest))");a=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='code');ns=dict(remote=remote,expected=expected,digest=digest);exec(compile(ast.Module(body=[a],type_ignores=[]),'fixed-placeholder','exec'),ns);code=ns['code'];compile(code,'actual-remote-deployment-command','exec')
def save(n,x):
 with (P/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
save('deployment-recovery-intent.json',{'at':time.time(),'original_failure':'CPU remote SyntaxError from broad placeholder replacement; no Python command body executed','generated_command_compile_pass':True,'actual_code_SHA':hashlib.sha256(code.encode()).hexdigest(),'only_existing_archive_SHA_then_missing_only_unpack':True,'no_reupload_GPU':True})
c=Client();c.initialize();r=c.request('tools/call',{'name':'run_command','arguments':dict(mount_id=M,command='python3 -c '+shlex.quote(code),env={'CUDA_VISIBLE_DEVICES':''},timeout_seconds=20)});save('actual-deployment-recovered.json',r);assert not r.get('isError'),r;o=r.get('structuredContent') or json.loads(r['content'][0]['text']);assert o['exit_code']==0,o;print(o['stdout'])
