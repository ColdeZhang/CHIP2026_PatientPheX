from pathlib import Path
import sys,json,shlex
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
E=Path(__file__).parent;P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';M='5a9955da-5240-41cc-9e52-05d3d8da8a3c'
code="""from pathlib import Path
import json,hashlib,zipfile,os
D=Path(%r);r=json.loads((D/'run/child-result.json').read_text());assert r['returncode']==0 and not r['timed_out'] and r['complete_exists'] and r['normalization_complete_exists'];native=json.loads((D/'run/native/complete.json').read_text());assert native['status']=='COMPLETE_FROZEN_PCL_F0_RAW' and native['all_documents']==16 and native['all_patients']==47;loaded=json.loads((D/'run/native/loaded.json').read_text());assert loaded['all206_cpu_values_exact'] and loaded['no_fit'];members=[]
for folder in ['run','source','input']:members.extend(p for p in (D/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pt')
for name in ['deployment.zip','deployment-complete.json','cpu/complete.json']:members.append(D/name)
inv={str(p.relative_to(D)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(members))};(D/'terminal-inventory.json').write_text(json.dumps(inv,indent=2)+'\\n');target=D/'terminal-evidence.zip';assert not target.exists()
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for n in inv:z.write(D/n,n)
 z.write(D/'terminal-inventory.json','terminal-inventory.json')
with target.open('rb') as f:os.fsync(f.fileno())
fd=os.open(D,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
info={'status':'COMPLETE_PCL_F0_FROZEN_ARCHIVE','path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'members':len(inv),'checkpoint':loaded['checkpoint'],'checkpoint_loaded_tensors':206,'model_weights_downloaded':False,'all16_raw_included':len(list((D/'run/native').glob('raw-*.json.gz')))==16,'encoded_payload_retained_remote_not_in_archive':json.loads((D/'cpu/complete.json').read_text())['files']['held']};assert info['all16_raw_included'];(D/'terminal-ready.json').write_text(json.dumps(info,indent=2)+'\\n');print(json.dumps(info))
"""%(P+'/research/patient-competition-calibration-v1')
c=Client();c.initialize();v=c.request('tools/call',{'name':'run_command','arguments':{'mount_id':M,'command':'python3 -c '+shlex.quote(code),'timeout_seconds':60}});(E/'pack-terminal-response.json').write_text(json.dumps(v,indent=2));assert not v.get('isError'),v;x=v.get('structuredContent') or json.loads(v['content'][0]['text']);assert x['exit_code']==0,x;r=json.loads(x['stdout']);(E/'terminal-ready.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
