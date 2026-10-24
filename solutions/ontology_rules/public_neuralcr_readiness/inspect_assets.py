"""CPU-only safe archive/shape inventory. Does not import TF/fastText or load pickle."""
from pathlib import Path, PurePosixPath
import json, hashlib, tarfile, mmap, struct, math, os, time, traceback
R=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def dump(name,x): (R/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
assert not (R/'shape-inventory.json').exists()
t=time.monotonic(); report={'status':'running','source_sha256':sha(Path(__file__)),'no_TF_fastText_model_or_pickle_load':True}
try:
 complete=json.loads((R/'downloads/complete.json').read_text()); assert complete['status']=='complete_official_assets_not_loaded'; report['download_complete']=complete
 completed={a['filename']:a for a in complete['outputs']}
 plan=json.loads((R/'download-plan.json').read_text())
 assets={}
 for a in plan['assets']:
  p=R/'downloads'/a['filename']; assert p.stat().st_size==a['bytes']; assets[a['filename']]={'bytes':p.stat().st_size,'sha256':sha(p)}
  assert assets[a['filename']]['sha256']==completed[a['filename']]['sha256'] and a['bytes']==completed[a['filename']]['bytes']
 report['complete_asset_files']=assets
 selected={'HP/ncr_weights.h5','HP/config.json','HP/onto.json'}
 entries=[]; extracted=[]; target=R/'extracted'; target.mkdir(exist_ok=False)
 with tarfile.open(R/'downloads/HP.tar.gz','r:gz') as tar:
  for member in tar:
   q=PurePosixPath(member.name); assert not q.is_absolute() and '..' not in q.parts
   assert member.isdir() or member.isfile(), ('unsupported_archive_member_type',member.name)
   entries.append({'name':member.name,'bytes':member.size,'type':'directory' if member.isdir() else 'file','extracted':member.name in selected})
   if member.name in selected:
    assert member.isfile(); dest=target/q; dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists(); h=hashlib.sha256(); count=0
    with tar.extractfile(member) as src,dest.open('xb') as out:
     for b in iter(lambda:src.read(1048576),b''): out.write(b);h.update(b);count+=len(b)
    assert count==member.size;extracted.append({'path':str(dest.relative_to(R)),'bytes':count,'sha256':h.hexdigest()})
 report['archive_members']=entries;report['extracted']=extracted
 hp=target/'HP'; assert (hp/'config.json').exists() and (hp/'ncr_weights.h5').exists()
 report['config']=json.loads((hp/'config.json').read_text())
 report['safe_ontology_JSON_available']=(hp/'onto.json').exists()
 if (hp/'onto.json').exists():
  onto=json.loads((hp/'onto.json').read_text()); report['ontology_JSON']={'keys':list(onto),'concepts':len(onto['concepts']),'root_id':onto.get('root_id'),'sparse_ancestor_edges':len(onto.get('sparse_ancestors',[]))}
 import h5py
 datasets=[]; seen=set()
 with h5py.File(hp/'ncr_weights.h5','r') as h:
  def visit(name,obj):
   if not isinstance(obj,h5py.Dataset):return
   addr=h5py.h5o.get_info(obj.id).addr
   if addr in seen:return
   seen.add(addr);datasets.append({'name':name,'shape':list(obj.shape),'dtype':str(obj.dtype),'unique_parameters':math.prod(obj.shape),'object_address':int(addr)})
  h.visititems(visit)
  report['h5_root_attributes']={k:[x.decode() if isinstance(x,bytes) else str(x) for x in v] if getattr(v,'ndim',0)>0 else str(v) for k,v in h.attrs.items()}
 report['H5_unique_datasets']=datasets;report['H5_unique_parameters']=sum(x['unique_parameters'] for x in datasets)
 p=R/'downloads/pmc_model_new.bin'
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
 report['status']='failed_preserved';report['error']=repr(e);report['traceback']=traceback.format_exc();raise
finally:
 report['elapsed_seconds']=time.monotonic()-t;dump('shape-inventory.json',report)
