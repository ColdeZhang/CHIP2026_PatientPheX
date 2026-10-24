from pathlib import Path
import base64,concurrent.futures,hashlib,json,shlex,sys,time,datetime
R=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
mount='476f81cf-09e7-445b-a7f0-d0e12b12e50f';remote='public-neuralcr-readiness/network-census/transfer-probe'
assert not (R/'transfer-probe.json').exists()
items=[]
for name in ['HP.tar.gz','pmc_model_new.bin']:
 data=(R/'local/output'/(name+'.part')).read_bytes();assert len(data)==1048576
 for offset in [0,524288]:
  part=data[offset:offset+524288];items.append({'name':name+'.'+str(offset)+'.bin','bytes':len(part),'sha256':hashlib.sha256(part).hexdigest(),'data':part})
def put(item):
 c=Client();c.initialize();t=time.monotonic()
 r=c.request('tools/call',{'name':'write_file','arguments':{'mount_id':mount,'path':remote+'/'+item['name'],'content':base64.b64encode(item['data']).decode(),'encoding':'base64','create_parents':True}});assert not r.get('isError'),r
 return {'name':item['name'],'seconds':time.monotonic()-t,'result':r['structuredContent']}
start=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:receipts=list(executor.map(put,items))
write_wall=time.monotonic()-start
expected={i['name']:{k:i[k] for k in ['bytes','sha256']} for i in items}
c=Client();c.initialize();code="from pathlib import Path;import hashlib,json;E="+repr(expected)+";A={n:{'bytes':Path(n).stat().st_size,'sha256':hashlib.sha256(Path(n).read_bytes()).hexdigest()} for n in E};assert A==E;print(json.dumps(A))"
r=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':mount,'command':'../../public-phenotagger-reuse/.venv/bin/python -c '+shlex.quote(code),'cwd':remote,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':15}})
assert not r.get('isError'),r;z=r['structuredContent']
# This directory is three levels below project root, so use python3 if the reused venv path is absent.
if z['exit_code']!=0:
 r=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':mount,'command':'python3 -c '+shlex.quote(code),'cwd':remote,'env':{'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':''},'timeout_seconds':15}});assert not r.get('isError'),r;z=r['structuredContent']
assert z['exit_code']==0,z
report={'status':'pass_4x512KiB_public_payload_remote_exact','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'max_workers':4,'total_bytes':2097152,'write_wall_seconds_including_client_init':write_wall,'overall_wall_seconds_with_remote_SHA':time.monotonic()-start,'remote_actual':json.loads(z['stdout']),'receipts':receipts,'projected_1751739725_seconds_at_write_wall':1751739725/2097152*write_wall,'projection_not_a_bulk_guarantee':True,'no_tunnel':True,'public_assets_only':True}
(R/'transfer-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
