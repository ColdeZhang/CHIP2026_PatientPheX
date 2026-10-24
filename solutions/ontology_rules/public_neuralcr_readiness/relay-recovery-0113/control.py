"""One inventory or bounded MCP-only recovery; never makes a public download request."""
from pathlib import Path
import argparse,base64,concurrent.futures,datetime,hashlib,json,math,shlex,subprocess,sys,threading,time
R=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
from mcp_client import Client
plan=json.loads((R/'recovery-plan.json').read_text())
remote=plan['remote_root'];absolute='/home/zhangyuheng/projects/CHIP2026_PatientPheX/'+remote
parser=argparse.ArgumentParser();parser.add_argument('action',choices=['inventory','recover','recover-inner','collect']);a=parser.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def write(p,o):p.write_text(json.dumps(o,indent=2)+'\n')
deadline=datetime.datetime.fromisoformat(plan['absolute_deadline']).timestamp()
local=threading.local()
def call(name,args):
    assert time.time()<deadline, 'original absolute deadline'
    if not hasattr(local,'client'):local.client=Client();local.client.initialize()
    x=local.client.request('tools/call',{'name':name,'arguments':{'mount_id':plan['mount_id'],**args}})
    if x.get('isError'):raise RuntimeError('MCP_tool_error:'+str(x.get('content'))[:400])
    return x['structuredContent']
def put(path,b):
    assert len(b)<=524288
    x=call('write_file',{'path':path,'content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True})
    assert x['sha256']==sha(b) and x['size']==len(b)
def fetch(name):
    x=call('read_file',{'path':remote+'/'+name,'offset':0,'limit':524288})
    b=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();(R/name).write_bytes(b);return b
if a.action=='inventory':
    assert not (R/'inventory-tool-receipt.json').exists()
    assert not Path('/proc/80372').exists(), 'Original writer alive'
    for n in ['recovery-plan.json','expected-segments.json','remote_inventory.py','remote_finalize.py','remote_launch.py']:
        put(remote+'/'+n,(R/n).read_bytes())
    command='timeout --signal=TERM --kill-after=15s 120s '+shlex.quote(plan['python'])+' remote_inventory.py'
    result=call('run_command',{'command':command,'cwd':remote,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'','OMP_NUM_THREADS':'1'},'timeout_seconds':135})
    write(R/'inventory-tool-receipt.json',result)
    report=json.loads(fetch('remote-inventory.json'));assert result['exit_code']==0 and report['status']=='complete_readonly_inventory'
    assert report['expected_manifest_sha256']==plan['expected_manifest_sha256'] and not report['mismatched']
    lookup=read(R/'local-lookup.json');assert all(str(s['start']) in lookup for s in report['missing']), 'Cannot recover remote-only prefix'
    write(R/'recovery-bound-plan.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_plan_sha256':sha((R/'recovery-plan.json').read_bytes()),'inventory_sha256':sha((R/'remote-inventory.json').read_bytes()),'missing_segments':report['missing'],'missing_bytes':sum(s['bytes'] for s in report['missing']),'absolute_deadline':plan['absolute_deadline'],'workers_max':8,'attempts_per_piece':3,'no_public_network_download':True})
    print(json.dumps({'inventory':'pass','segments_verified':report['verified_segments'],'missing_segments':len(report['missing']),'missing_bytes':sum(s['bytes'] for s in report['missing'])}));raise SystemExit()
if a.action=='recover':
    assert not (R/'recovery-local-job.json').exists() and not Path('/proc/80372').exists()
    seconds=math.floor(deadline-time.time());assert seconds>150
    command=['timeout','--signal=TERM','--kill-after=30s',str(seconds)+'s',sys.executable,str(Path(__file__)),'recover-inner']
    with (R/'stdout.log').open('xb') as out,(R/'stderr.log').open('xb') as err:p=subprocess.Popen(command,stdout=out,stderr=err,start_new_session=True)
    write(R/'recovery-local-job.json',{'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'command':command,'remaining_cap_seconds':seconds,'absolute_deadline':plan['absolute_deadline'],'source_sha256':sha(Path(__file__).read_bytes()),'bound_plan_sha256':sha((R/'recovery-bound-plan.json').read_bytes())})
    print(json.dumps({'pid':p.pid,'cap_seconds':seconds}));raise SystemExit()
if a.action=='collect':
    job=read(R/'assembly-job.json');status=call('read_process_output',{'process_id':job['id'],'stream':'stderr','cursor':0,'limit':4096});write(R/'assembly-process-status.json',status)
    if status['running']:print(json.dumps({'assembly_running':True}));raise SystemExit()
    terminal=json.loads(fetch('pmc_model_new.bin.segments.json.job.json'));report=json.loads(fetch('pmc_model_new.bin.assembly.json'))
    assert terminal['returncode']==0 and report['status']=='complete_assembled_and_verified'
    parent=read(R/'prior-state.json');hp=parent['assets'][0]
    combined={'status':'complete_two_official_assets_verified_native_not_run','plan_sha256':parent['plan_sha256'],'assets':[hp,report],'original_attempt3_failed_preserved':{'path':str(R/'prior-state.json'),'sha256':sha((R/'prior-state.json').read_bytes())},'recovery_bound_plan_sha256':sha((R/'recovery-bound-plan.json').read_bytes()),'original_deadline':plan['absolute_deadline']}
    write(R.parent/'postdownload-readiness/verified-assets.json',combined);print(json.dumps({'complete':True,'FT_bytes':report['bytes'],'FT_sha256':report['sha256'],'assembly_seconds':report['elapsed_seconds']}));raise SystemExit()

assert a.action=='recover-inner' and not (R/'recovery-state.json').exists()
bound=read(R/'recovery-bound-plan.json');assert bound['parent_plan_sha256']==sha((R/'recovery-plan.json').read_bytes())
assert bound['inventory_sha256']==sha((R/'remote-inventory.json').read_bytes())
lookup=read(R/'local-lookup.json');expected=read(R/'expected-segments.json');success={};failures=[];lock=threading.Lock();started=time.monotonic()
state={'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bound_plan_sha256':sha((R/'recovery-bound-plan.json').read_bytes()),'no_public_network_download':True}
def save():
    state.update(successful_segments=list(success.values()),failures=failures,elapsed_seconds=time.monotonic()-started)
    write(R/'recovery-state.json',state)
save()
def transfer(s):
    loc=lookup[str(s['start'])]
    with Path(loc['local_path']).open('rb') as f:f.seek(loc['local_offset']);b=f.read(s['bytes'])
    assert len(b)==s['bytes'] and sha(b)==s['sha256'];name=f"pieces/pmc_model_new.bin.{s['start']}-{s['start']+s['bytes']}.bin";errors=[]
    for attempt in range(3):
        try:
            assert time.time()+130<deadline, 'assembly budget reserve'
            put(remote+'/'+name,b)
            with lock:success[s['start']]={**s,'path':absolute+'/'+name};save()
            return
        except Exception as e:errors.append(str(e))
    with lock:failures.append({'segment':s,'errors':errors});save()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futures=[pool.submit(transfer,s) for s in bound['missing_segments']]
    for f in concurrent.futures.as_completed(futures):f.result()
if failures:state['status']='failed_fragments_preserved';save();raise SystemExit(2)
expected['segments']=[success.get(s['start'],s) for s in expected['segments']]
payload=(json.dumps(expected,separators=(',',':'))+'\n').encode();assert len(payload)<=524288
(R/'pmc_model_new.bin.segments.json').write_bytes(payload);put(remote+'/pmc_model_new.bin.segments.json',payload)
request={'command':shlex.quote(plan['python'])+' remote_launch.py pmc_model_new.bin.segments.json','cwd':remote,'env':{'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}}
write(R/'assembly-start-intent.json',request)
try:job=call('start_process',request)
except Exception as error:
    state.update(status='assembly_start_outcome_unknown_do_not_repeat',error=str(error));save();raise
write(R/'assembly-job.json',job);state['status']='all_missing_pieces_written_assembly_started';save();print(json.dumps(job))
