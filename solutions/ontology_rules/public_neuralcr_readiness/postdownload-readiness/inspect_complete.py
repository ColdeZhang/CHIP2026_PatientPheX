"""Full fastText exact dictionary/matrix inventory; reuse actual full HP H5 report."""
from pathlib import Path
import os,sys,json,hashlib,mmap,struct,time,traceback
R=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (R/'shape-inventory.json').exists()
plan=json.loads((R/'shape-plan.json').read_text());t=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
report={'status':'running','source_sha256':sha(__file__),'plan_sha256':sha(R/'shape-plan.json'),'no_model_or_NN':True}
try:
 for name,ref in plan['inputs'].items():assert Path(ref['path']).stat().st_size==ref['bytes'] and sha(ref['path'])==ref['sha256'],name
 hp=json.loads(Path(plan['HP_report']).read_text());assert hp['status']=='complete_assembled_and_verified'
 report.update(H5_unique_parameters=hp['H5_unique_parameters'],H5_unique_datasets=hp['H5_unique_datasets'],safe_ontology_JSON_available=True,ontology_JSON={'concepts':hp['ontology_concepts']},HP_actual=hp)
 p=Path(plan['fastText_file'])
 with p.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
  magic,version=struct.unpack_from('<ii',mm,0); args=struct.unpack_from('<12i',mm,8); dim=args[0];bucket=args[8]
  size,nwords,nlabels,ntokens=struct.unpack_from('<iiiq',mm,64);prune=struct.unpack_from('<q',mm,84)[0]
  assert magic==793712314 and version==12 and (size,nwords,nlabels)==(666817,666817,0)
  ft={'magic':magic,'version':version,'dimension':dim,'bucket':bucket,'vocabulary':nwords,'labels':nlabels,'token_count':ntokens,'standard_layout_pruneidx_size':prune,'standard_layout_valid':prune>=-1,'dictionary_count':0}
  # This observes the actual legacy header. No bytes or model are modified here.
  if prune < -1:
   end=mm.find(b'\0',84,4096); assert end==87
   observed_prune=struct.unpack_from('<q',mm,end+1)[0]; assert observed_prune==-1
   firstword=bytes(mm[84:end]).decode('utf8'); count,kind=struct.unpack_from('<qb',mm,end+9)
   assert count>0 and kind==0; pos=end+18; words=1
   ft['observed_legacy_prefix']={'firstword':firstword,'word_bytes_including_NUL':4,'pruneidx_size':observed_prune,'pruneidx_offset':88,'first_count':count,'first_type':kind,'remaining_entries_begin':pos}
  else: pos=92;words=0
  while words<size:
   end=mm.find(b'\0',pos,min(pos+65536,len(mm)));assert end>=pos
   count,kind=struct.unpack_from('<qb',mm,end+1); assert count>=0 and kind==0
   pos=end+10;words+=1
  ft['dictionary_count']=words;ft['after_dictionary_offset']=pos
  matrices=[]
  for label in ('input_word_and_subword','output'):
   quant=mm[pos];rows,cols=struct.unpack_from('<qq',mm,pos+1);assert quant==0 and cols==100
   start=pos+17;end=start+rows*cols*4;assert end<=len(mm)
   matrices.append({'name':label,'quantized':False,'flag_offset':pos,'header_offset':pos+1,'shape':[rows,cols],'dtype':'float32','parameters':rows*cols,'data_offset':start,'data_end_exclusive':end})
   pos=end
  assert pos==len(mm);assert matrices[0]['shape']==[nwords+bucket,dim] and matrices[1]['shape']==[nwords,dim]
  ft['matrices']=matrices;ft['unique_parameters_including_output']=sum(x['parameters'] for x in matrices);ft['EOF_exact']=True
 report['fastText']=ft
 report['NCR_total_unique_parameters']=ft['unique_parameters_including_output']+report['H5_unique_parameters']
 current=9597467990;total=report['NCR_total_unique_parameters'];report['competition_budget']={'current_production':current,'limit':10000000000,'append_total':current+total,'append_fits':current+total<=10000000000,'replace_BioBERT_total':current-121587826+total,'replace_only_added_BioBERT_fits':current-121587826+total<=10000000000}
 report['native_load_preconditions']={'safe_JSON_ontology':report['safe_ontology_JSON_available'],'fastText_standard_header':ft['standard_layout_valid'],'CPU_load_not_run':True}
 report['status']='complete_inventory'
except Exception as e:
 report.update(status='failed_preserved',error_type=type(e).__name__,traceback=traceback.format_exc());raise
finally:
 report['elapsed_seconds']=time.monotonic()-t;(R/'shape-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
