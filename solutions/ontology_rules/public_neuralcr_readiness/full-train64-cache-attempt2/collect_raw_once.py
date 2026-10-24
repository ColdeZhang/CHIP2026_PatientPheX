"""One bounded read of the complete small raw-spans gzip after actual link recovery."""
from pathlib import Path
import base64,datetime,hashlib,json,sys,time
R=Path(__file__).resolve().parent;final=json.loads((R/'finalization-result.json').read_text());assert final['status']=='complete_cache_verified_by_CPU_finalization_original_GPU_rc1_preserved';expected=final['outputs']['native-raw-spans.jsonl.gz'];assert expected['bytes']==192972 and expected['sha256']=='efdf3bebcd6114c1803cf7db57ef7b441c05e6a89b21ce687b124cb43c5ef175';assert not(R/'raw-once-intent.json').exists();start=time.monotonic();sha=lambda b:hashlib.sha256(b).hexdigest()
request={'mount_id':'476f81cf-09e7-445b-a7f0-d0e12b12e50f','path':expected['path'].removeprefix('/home/zhangyuheng/projects/CHIP2026_PatientPheX/'),'offset':0,'limit':expected['bytes']};intent={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(Path(__file__).read_bytes()),'request':request,'expected':expected,'remote_full_cache_finalization_sha256':sha((R/'finalization-result.json').read_bytes()),'no_remote_writes_or_NN':True};(R/'raw-once-intent.json').write_text(json.dumps(intent,indent=2)+'\n');report={**intent,'status':'reading_only_small_raw_spans'}
try:
 sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
 c=Client();c.initialize();x=c.request('tools/call',{'name':'read_file','arguments':request})
 if x.get('isError'):raise RuntimeError(str(x.get('content'))[:700])
 x=x['structuredContent'];b=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();assert len(b)==expected['bytes'] and sha(b)==expected['sha256'];target=R/'recovered/native-raw-spans.jsonl.gz';target.parent.mkdir(exist_ok=True)
 if target.exists():assert target.read_bytes()==b
 else:
  with target.open('xb') as f:f.write(b)
 report.update(status='complete_local_raw_spans_remote_full_cache_proof',verified={'native-raw-spans.jsonl.gz':{'local_path':str(target),'bytes':len(b),'sha256':sha(b)}},full_top1_cache_locally_required=False,head_locally_required=False)
except BaseException as e:report.update(status='small_raw_spans_read_failed_preserved',error=repr(e));raise
finally:
 report['elapsed_seconds']=time.monotonic()-start;(R/'raw-collection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report.get(k) for k in ['status','elapsed_seconds','error']}))
