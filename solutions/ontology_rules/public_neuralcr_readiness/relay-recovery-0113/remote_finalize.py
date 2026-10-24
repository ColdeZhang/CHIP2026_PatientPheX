"""Assemble only exact verified ranges; HP-only safe shape gate, no TF/NN."""
from pathlib import Path, PurePosixPath
import sys,json,hashlib,tarfile,math,time,os,traceback
R=Path(__file__).resolve().parent

def run(manifest):
 t=time.monotonic(); report={'status':'running','no_NN':True}
 def save(): (R/(manifest['filename']+'.assembly.json')).write_text(json.dumps(report,indent=2)+'\n')
 try:
  assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
  dest=R/'downloads'/manifest['filename'];dest.parent.mkdir(exist_ok=True);assert not dest.exists()
  position=0; totalsha=hashlib.sha256(); checked=[]
  with dest.with_suffix(dest.suffix+'.assembling').open('xb') as output:
   for s in sorted(manifest['segments'],key=lambda x:x['start']):
    assert s['start']==position and s['bytes']>0
    p=Path(s['path']);p=p if p.is_absolute() else Path(manifest['segment_root'])/p;assert p.stat().st_size==s['bytes']; h=hashlib.sha256(); count=0
    with p.open('rb') as f:
     for b in iter(lambda:f.read(1048576),b''):output.write(b);h.update(b);totalsha.update(b);count+=len(b)
    assert count==s['bytes'] and h.hexdigest()==s['sha256'];position+=count
    checked.append({'start':s['start'],'bytes':count,'sha256':h.hexdigest()})
  assert position==manifest['expected_bytes'];dest.with_suffix(dest.suffix+'.assembling').rename(dest)
  report.update({'filename':manifest['filename'],'path':str(dest),'bytes':position,'sha256':totalsha.hexdigest(),'segments_checked':len(checked)})
  if manifest['filename']=='HP.tar.gz':
   target=R/'extracted';target.mkdir(exist_ok=False); selected={'HP/ncr_weights.h5','HP/config.json','HP/onto.json'}; entries=[]; extracted=[]
   with tarfile.open(dest,'r:gz') as tar:
    for member in tar:
     q=PurePosixPath(member.name);assert not q.is_absolute() and '..' not in q.parts
     assert member.isdir() or member.isfile()
     entries.append({'name':member.name,'bytes':member.size,'extracted':member.name in selected})
     if member.name in selected:
      assert member.isfile(); p=target/q;p.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
      with tar.extractfile(member) as src,p.open('xb') as f:
       for b in iter(lambda:src.read(1048576),b''):f.write(b);h.update(b);n+=len(b)
      assert n==member.size;extracted.append({'path':str(p),'bytes':n,'sha256':h.hexdigest()})
   assert {x['name'] for x in entries if x['extracted']}==selected
   hp=target/'HP';import h5py
   seen=set(); arrays=[]
   with h5py.File(hp/'ncr_weights.h5','r') as h:
    def visit(n,o):
     if isinstance(o,h5py.Dataset):
      address=h5py.h5o.get_info(o.id).addr
      if address not in seen:
       seen.add(address);arrays.append({'name':n,'shape':list(o.shape),'dtype':str(o.dtype),'parameters':math.prod(o.shape),'address':int(address)})
    h.visititems(visit)
   count=sum(x['parameters'] for x in arrays);onto=json.loads((hp/'onto.json').read_text());projected=count+333363400
   report.update({'archive_members':entries,'extracted':extracted,'config':json.loads((hp/'config.json').read_text()),'ontology_concepts':len(onto['concepts']),'H5_unique_datasets':arrays,'H5_unique_parameters':count,'fastText_header_projected_parameters':333363400,'fastText_complete_inventory_pending':True,'NCR_conservative_projected_parameters':projected,'append_total':9597467990+projected,'replace_BioBERT_total':9597467990-121587826+projected,'continue_fastText':9597467990-121587826+projected<=10000000000})
  report['status']='complete_assembled_and_verified';return report
 except Exception as e: report.update(status='failed_preserved',error_type=type(e).__name__,traceback=traceback.format_exc());raise
 finally: report['elapsed_seconds']=time.monotonic()-t;save()
if __name__=='__main__':run(json.loads(Path(sys.argv[1]).read_text()))
