"""Read only complete compact cache artifacts with bounded 64KiB transfers and full SHA."""
from pathlib import Path
import argparse,base64,datetime,gzip,hashlib,json,os,re,sys,time
R=Path(__file__).resolve().parent;L=R/'recovered';L.mkdir(exist_ok=True);state=json.loads((R/'run-state.json').read_text());result=json.loads((R/'finalization-result.json').read_text());assert state['status']=='failed_preserved' and state['returncode']==1 and result['status']=='complete_cache_verified_by_CPU_finalization_original_GPU_rc1_preserved' and json.loads((R/'finalization-job.json').read_text())['returncode']==0
ap=argparse.ArgumentParser();ap.add_argument('--local-input',required=True);ap.add_argument('--chunk-bytes',type=int,default=65536,choices=[16384,65536]);args=ap.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest();assert sha((R/'cache-result.json').read_bytes())==result['original_failed_result_still_sha256'];started=time.monotonic();report={'status':'collecting_read_only','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(Path(__file__).read_bytes()),'result_sha256':sha((R/'finalization-result.json').read_bytes()),'verified':{},'chunk_limit':args.chunk_bytes,'attempt_wall_cap_seconds':180,'NN_fit_quality':False}
def save():report['elapsed_seconds']=time.monotonic()-started;(R/'collection-state.json').write_text(json.dumps(report,indent=2)+'\n')
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
c=Client();c.initialize();save()
try:
 for name in ['top1.npy.gz','columns.json','native-raw-spans.jsonl.gz','batch-progress.jsonl']:
  ref=result['outputs'][name];assert ref['path'].startswith('/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-neuralcr-readiness/');target=L/name;part=L/(name+'.part');assert 0<ref['bytes']<16777216
  if target.exists():assert target.stat().st_size==ref['bytes'] and sha(target.read_bytes())==ref['sha256']
  else:
   offset=part.stat().st_size if part.exists() else 0;assert offset<=ref['bytes']
   with part.open('ab') as out:
    while offset<ref['bytes']:
     assert time.monotonic()-started<180,'bounded read-only attempt wall cap reached'
     amount=min(args.chunk_bytes,ref['bytes']-offset);report['active']={'name':name,'offset':offset,'remaining':ref['bytes']-offset};save()
     x=c.request('tools/call',{'name':'read_file','arguments':{'mount_id':'476f81cf-09e7-445b-a7f0-d0e12b12e50f','path':ref['path'].removeprefix('/home/zhangyuheng/projects/CHIP2026_PatientPheX/'),'offset':offset,'limit':amount}})
     if x.get('isError'):raise RuntimeError(str(x.get('content'))[:700])
     x=x['structuredContent'];b=base64.b64decode(x['content']) if x.get('encoding')=='base64' else x['content'].encode();assert len(b)==amount;out.write(b);out.flush();os.fsync(out.fileno());offset+=len(b)
   assert part.stat().st_size==ref['bytes'] and sha(part.read_bytes())==ref['sha256'];part.rename(target)
  report['verified'][name]={'local_path':str(target),'bytes':ref['bytes'],'sha256':ref['sha256']};save()
 canonical=gzip.decompress((L/'top1.npy.gz').read_bytes());ref=result['outputs']['top1.npy'];assert len(canonical)==ref['bytes'] and sha(canonical)==ref['sha256'];p=L/'top1.npy'
 if p.exists():assert p.read_bytes()==canonical
 else:
  with p.open('xb') as out:out.write(canonical)
 report['verified']['top1.npy']={'local_path':str(p),'bytes':len(canonical),'sha256':sha(canonical),'source':'verified_lossless_gzip'}
 q=R.parent/'input-only-census/unique-normalized-queries.jsonl';ref=result['outputs']['queries.jsonl'];assert q.stat().st_size==ref['bytes'] and sha(q.read_bytes())==ref['sha256'];report['verified']['queries.jsonl']={'local_path':str(q),'bytes':q.stat().st_size,'sha256':ref['sha256'],'source':'existing_full_exact_local_file_no_transfer'}
 import numpy as np
 a=np.load(p,allow_pickle=False);assert a.shape==(408272,) and a.dtype==np.dtype([('column','<u2'),('score','<f4')],align=False) and np.isfinite(a['score']).all() and ((a['score']>=0)&(a['score']<=1)).all() and (a['column']<14261).all()
 plan=json.loads((R/'plan.json').read_text());bound=json.loads((R/'bound-plan.json').read_text());input_path=Path(args.local_input);input_bytes=input_path.read_bytes();assert len(input_bytes)==plan['input']['bytes'] and sha(input_bytes)==plan['input']['sha256'];docs={d['pmc_id']:d for d in map(json.loads,input_bytes.splitlines())};assert len(docs)==64
 labels=json.loads((L/'columns.json').read_text())['labels'];assert len(labels)==14261 and sha('\n'.join(labels).encode())==bound['columns_order_sha256'];keys=[json.loads(x)['query'] for x in q.read_text().splitlines()]
 native=R.parent/'postdownload-readiness/ncrmodel.py';assert sha(native.read_bytes())==plan['native_source']['sha256'];from prepare_inputs import extract_native
 _,tokenize,_=extract_native(native.read_text());span_count=0;unique=set()
 with gzip.open(L/'native-raw-spans.jsonl.gz','rt',encoding='utf-8') as f:
  for line in f:
   z=json.loads(line);d=docs[z['pmc_id']];passage=d['full_text'][z['passage_index']];start=z['local_start'];end=z['local_end'];assert 0<=start<end<=len(passage['text']) and z['text']==passage['text'][start:end] and z['passage_offset']==passage['offset'] and z['offset']==passage['offset']+start and z['length']==end-start and z['passage_type']==passage['type']
   index=z['cache_row'];assert 0<=index<408272;stored=a[index];assert labels[int(stored['column'])]==z['raw_hpo_id'] and int(stored['score'].view(np.uint32))==z['score_float32_bits']==int(np.float32(z['score']).view(np.uint32))
   key=' '.join(tokenize(re.sub(r'[\\/\r\n\t-]',' ',z['text']))[:49]);assert key==keys[index];identity=(z['pmc_id'],z['passage_index'],start,end,z['raw_hpo_id']);assert identity not in unique;unique.add(identity);span_count+=1
 assert span_count==result['native_decode']['raw_native_spans']
 progress=[json.loads(x) for x in (L/'batch-progress.jsonl').read_text().splitlines()];assert len(progress)==798
 for i,z in enumerate(progress):assert z['batch']==i and z['start']==i*512 and z['end']==min((i+1)*512,408272) and z['rows']==z['end']-z['start'] and z['origin']==('verified_probe_reuse' if i<3 else 'new_NN') and z['score_float32_bits_preserved'] is True
 report['local_artifact_contract']={'all_raw_spans':span_count,'raw_text_offsets_query_row_HPO_score_bits_exact':True,'all798_batch_intervals_cover408272_once':True,'no_threshold_redecode_or_model_call':True,'input_sha256':sha(input_bytes)}
 report.update(status='complete_all_compact_artifacts_local_full_SHA',finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),canonical_shape=list(a.shape),canonical_dtype=a.dtype.descr,local_numpy_version=np.__version__,no_threshold_redecode_on_different_numpy=True);report.pop('active',None);save();print(json.dumps({'status':report['status'],'bytes_received':sum(v['bytes'] for k,v in report['verified'].items() if k not in ['queries.jsonl','top1.npy']),'seconds':report['elapsed_seconds']}))
except BaseException as e:report.update(status='collection_incomplete_preserved',error=repr(e));save();raise
