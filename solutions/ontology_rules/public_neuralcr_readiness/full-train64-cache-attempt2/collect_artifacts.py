"""Read only complete compact cache artifacts with bounded 64KiB transfers and full SHA."""
from pathlib import Path
import base64,datetime,gzip,hashlib,json,os,sys,time
R=Path(__file__).resolve().parent;state=json.loads((R/'run-state.json').read_text());result=json.loads((R/'cache-result.json').read_text());assert state['status']=='complete_cache_GPU_released' and result['status']=='complete_all_keys_top1_and_native_raw_span_cache_no_quality'
sha=lambda b:hashlib.sha256(b).hexdigest();assert sha((R/'cache-result.json').read_bytes())==state['result_sha256'];started=time.monotonic();report={'status':'collecting_read_only','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(Path(__file__).read_bytes()),'result_sha256':state['result_sha256'],'verified':{},'chunk_limit':65536,'NN_fit_quality':False}
def save():report['elapsed_seconds']=time.monotonic()-started;(R/'collection-state.json').write_text(json.dumps(report,indent=2)+'\n')
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();save()
try:
 for name in ['top1.npy.gz','columns.json','native-raw-spans.jsonl.gz','batch-progress.jsonl']:
  ref=result['outputs'][name];target=R/name;part=R/(name+'.part');assert 0<ref['bytes']<16777216
  if target.exists():assert target.stat().st_size==ref['bytes'] and sha(target.read_bytes())==ref['sha256']
  else:
   offset=part.stat().st_size if part.exists() else 0;assert offset<=ref['bytes']
   with part.open('ab') as out:
    while offset<ref['bytes']:
     amount=min(65536,ref['bytes']-offset);report['active']={'name':name,'offset':offset,'remaining':ref['bytes']-offset};save()
     x=c.request('tools/call',{'name':'read_file','arguments':{'mount_id':'476f81cf-09e7-445b-a7f0-d0e12b12e50f','path':'public-neuralcr-readiness/full-train64-cache-attempt2/'+name,'offset':offset,'limit':amount}})
     if x.get('isError'):raise RuntimeError(str(x.get('content'))[:700])
     x=x['structuredContent'];b=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();assert len(b)==amount;out.write(b);out.flush();os.fsync(out.fileno());offset+=len(b)
   assert part.stat().st_size==ref['bytes'] and sha(part.read_bytes())==ref['sha256'];part.rename(target)
  report['verified'][name]={'local_path':str(target),'bytes':ref['bytes'],'sha256':ref['sha256']};save()
 canonical=gzip.decompress((R/'top1.npy.gz').read_bytes());ref=result['outputs']['top1.npy'];assert len(canonical)==ref['bytes'] and sha(canonical)==ref['sha256'];p=R/'top1.npy'
 if p.exists():assert p.read_bytes()==canonical
 else:
  with p.open('xb') as out:out.write(canonical)
 report['verified']['top1.npy']={'local_path':str(p),'bytes':len(canonical),'sha256':sha(canonical),'source':'verified_lossless_gzip'}
 q=R.parent/'input-only-census/unique-normalized-queries.jsonl';ref=result['outputs']['queries.jsonl'];assert q.stat().st_size==ref['bytes'] and sha(q.read_bytes())==ref['sha256'];report['verified']['queries.jsonl']={'local_path':str(q),'bytes':q.stat().st_size,'sha256':ref['sha256'],'source':'existing_full_exact_local_file_no_transfer'}
 import numpy as np
 a=np.load(p,allow_pickle=False);assert a.shape==(408272,) and a.dtype==np.dtype([('column','<u2'),('score','<f4')],align=False) and np.isfinite(a['score']).all() and ((a['score']>=0)&(a['score']<=1)).all() and (a['column']<14261).all()
 report.update(status='complete_all_compact_artifacts_local_full_SHA',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),canonical_shape=list(a.shape),canonical_dtype=a.dtype.descr,local_numpy_version=np.__version__,no_threshold_redecode_on_different_numpy=True);report.pop('active',None);save();print(json.dumps({'status':report['status'],'bytes_received':sum(v['bytes'] for k,v in report['verified'].items() if k not in ['queries.jsonl','top1.npy']),'seconds':report['elapsed_seconds']}))
except BaseException as e:report.update(status='collection_incomplete_preserved',error=repr(e));save();raise
