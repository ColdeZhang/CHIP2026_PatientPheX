"""One conditional serialization repair; official asset untouched, matrix bytes exact."""
from pathlib import Path
import json,hashlib,mmap,struct,time,os
R=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dictionary(path,legacy):
 with path.open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as m:
  assert struct.unpack_from('<ii',m,0)==(793712314,12)
  size,nwords,nlabels,ntokens=struct.unpack_from('<iiiq',m,64);assert (size,nwords,nlabels)==(666817,666817,0)
  pos=84 if legacy else 92;h=hashlib.sha256();first=None
  if not legacy:assert struct.unpack_from('<q',m,84)[0]==-1
  for i in range(size):
   end=m.find(b'\0',pos,min(pos+65536,len(m)));assert end>=pos
   word=bytes(m[pos:end+1]);count_at=end+1
   if legacy and i==0:
    assert word==b'the\0' and count_at==88 and struct.unpack_from('<q',m,88)[0]==-1;count_at+=8
   count,kind=struct.unpack_from('<qb',m,count_at);assert count>=0 and kind==0
   h.update(word);h.update(m[count_at:count_at+9]);pos=count_at+9
   if i==0:first={'word_hex':word.hex(),'count':count,'kind':kind}
  matrices=[]
  for label in ['input_word_and_subword','output']:
   quant=m[pos];rows,cols=struct.unpack_from('<qq',m,pos+1);assert quant==0 and cols==100
   start=pos+17;end=start+rows*cols*4;assert end<=len(m)
   digest=hashlib.sha256()
   for off in range(start,end,1048576):digest.update(m[off:min(off+1048576,end)])
   matrices.append({'name':label,'shape':[rows,cols],'data_offset':start,'data_end_exclusive':end,'sha256':digest.hexdigest()});pos=end
  assert pos==len(m) and matrices[0]['shape']==[2666817,100] and matrices[1]['shape']==[666817,100]
  return {'word_count':size,'words_counts_types_ordered_sha256':h.hexdigest(),'first':first,'matrices':matrices,'EOF_exact':True}
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';t=time.monotonic();plan=json.loads((R/'repair-plan.json').read_text());assert not(R/'repair-report.json').exists()
 source=Path(plan['original_file']);assert source.stat().st_size==plan['original_bytes'] and sha(source)==plan['original_sha256']
 inventory=json.loads(Path(plan['inventory_path']).read_text());assert sha(Path(plan['inventory_path']))==plan['inventory_sha256']
 ft=inventory['fastText'];assert inventory['status']=='complete_inventory' and ft['EOF_exact'] and ft['dictionary_count']==666817 and not ft['standard_layout_valid']
 assert ft['observed_legacy_prefix']['pruneidx_size']==-1 and ft['observed_legacy_prefix']['pruneidx_offset']==88
 before=source.stat();original=dictionary(source,True);target=R/'pmc_model_new.standard-layout.bin';assert not target.exists()
 with source.open('rb') as src,target.open('xb') as dst:
  head=src.read(96);assert head[84:88]==b'the\0' and head[88:96]==b'\xff'*8;dst.write(head[:84]+head[88:96]+head[84:88]);tail=hashlib.sha256()
  for b in iter(lambda:src.read(1048576),b''):dst.write(b);tail.update(b)
 fixed=dictionary(target,False);assert fixed==original
 assert (before.st_size,before.st_mtime_ns)==(source.stat().st_size,source.stat().st_mtime_ns)
 # Full tail equality is stronger than just matching matrix checksums.
 with source.open('rb') as a,target.open('rb') as b:
  assert a.read(84)==b.read(84);a.seek(96);b.seek(96)
  for x in iter(lambda:a.read(1048576),b''):assert x==b.read(len(x))
  assert b.read()==b''
 result={'status':'pass_exact_12byte_layout_repair','source_sha256':sha(Path(__file__)),'plan_sha256':sha(R/'repair-plan.json'),'original':{'path':str(source),'bytes':before.st_size,'sha256':plan['original_sha256'],'unchanged':True},'derived':{'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target)},'change':'Only original bytes [84:88] (the+NUL) and [88:96] (prune=-1) exchange order. All bytes before84 and after96 exact.','ordered_dictionary_and_both_full_matrices':fixed,'all_numerical_parameter_bytes_unchanged':True,'parameter_inventory_unchanged':True,'no_Cpp_model_or_NN_load':True,'elapsed_seconds':time.monotonic()-t}
 (R/'repair-report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
