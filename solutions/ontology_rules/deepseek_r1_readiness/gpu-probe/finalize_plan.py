"""Pure local binding of actual asset terminal; no model, MCP, or start."""
from pathlib import Path
import hashlib,json,os
G=Path(__file__).parent;D=G.parent;A=D/'assets';remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/assets/'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=remote+p.name,bytes=p.stat().st_size,sha256=sha(p))
def main():
 complete=json.loads((A/'complete.json').read_text());header=json.loads((A/'actual-header-inventory.json').read_text());expected=json.loads((A/'expected-files.json').read_text())
 assert complete['status']=='ASSET_FILES_AND_ACTUAL_HEADERS_VERIFIED' and complete['MS_revision']=='5ee5fa4832e658ab84298806de06ed2a167834bd'
 assert len(complete['files'])==len(expected['files'])==9
 actual={x['name']:x for x in complete['files']}
 for e in expected['files']:
  x=actual[e['name']];assert x['sha256']==e['sha256'] and x['bytes']==e['bytes'] and x['status'] in ['REUSED_FULL_SHA_MATCH','DOWNLOADED_FULL_SHA_MATCH']
 assert complete['header_inventory_sha256']==sha(A/'actual-header-inventory.json') and header['tensor_keys']==399 and header['full_bundle_parameters']==8190735360
 assert complete['ended_epoch']<=complete['absolute_deadline_epoch']
 plan=json.loads((G/'plan.template.json').read_text());assert plan['execution_ready'] is False
 plan.update(execution_ready=True,status='ACTUAL_ASSETS_CPU_YARN_BOUND_UNIQUE_THREE_ROW_PROBE_READY_NO_GPU_STARTED',assets_result=ref(A/'complete.json'),header_inventory=ref(A/'actual-header-inventory.json'))
 plan['authorization']='Root authorized unique3fixedinputonlyrows after actualassets/nativeCPU/YaRN and freshGPU1; process180s+TERM30/max1024; firsttruncationorinvalidstops. No full80/gold/quality.'
 for n,h in plan['sources'].items():assert sha(G/n)==h
 with (G/'plan.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(status='FINAL_ACTUAL_PLAN_BOUND_NO_START',sha256=sha(G/'plan.json'),assets_sha256=sha(A/'complete.json'),header_sha256=sha(A/'actual-header-inventory.json'))))
if __name__=='__main__':main()
