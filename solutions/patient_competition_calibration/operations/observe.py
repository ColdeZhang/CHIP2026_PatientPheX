from pathlib import Path
import sys,json,shlex,time
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
D=Path(__file__).parent;P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';phase=sys.argv[1]
code="""from pathlib import Path
import json,os,hashlib
D=Path(%r);out={}
for name in ['started.json','child.json','child-result.json','fresh-inventory.json','released-gpu.json','native/initialized.json','native/updates.json','native/progress.json','native/complete.json','native/epochs.json','normalization/pre-normalization.json','normalization/complete.json']:
 p=D/name
 if p.exists():out[name]={'value':json.loads(p.read_text()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
for name in ['stderr.log','native.stderr.log','normalization.stderr.log']:
 p=D/name
 if p.exists():out[name+'_tail']=p.read_text()[-4500:]
print(json.dumps(out))
"""%(P+'/research/patient-competition-calibration-v1/'+phase)
c=Client();c.initialize();v=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':M,'command':'python3 -c '+shlex.quote(code),'timeout_seconds':30}});(D/(phase+'-observe-'+str(time.time_ns())+'.json')).write_text(json.dumps(v,indent=2));assert not v.get('isError'),v;x=json.loads((v.get('structuredContent') or json.loads(v['content'][0]['text']))['stdout']);dest=D/phase;dest.mkdir(exist_ok=True)
for k,v in x.items():
 if not k.endswith('_tail'):
  p=dest/k;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v['value'],indent=2)+'\n')
brief={k:v for k,v in x.items() if k in ['child.json','child-result.json','native/progress.json','released-gpu.json'] or k.endswith('_tail')}
if 'started.json' in x: brief['clock']={k:x['started.json']['value'][k] for k in ['started_utc','deadline','driver_pid']}
print(json.dumps(brief,indent=2))
if 'native/complete.json' in x: print('NATIVE_COMPLETE',x['native/complete.json']['value']['status'])
if 'normalization/complete.json' in x: print('NORMALIZATION_COMPLETE',x['normalization/complete.json']['value']['mapped_pairs'])
