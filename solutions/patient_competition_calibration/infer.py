"""Load the frozen final PCL model once; complete pure F0 inference, no fit."""
from pathlib import Path
import argparse,json,time,torch
from transformers import AutoModel
from common import ref,save,unpack,packed
from model import PatientSpanModel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--encoded',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text());O=Path(a.output);O.mkdir(parents=True,exist_ok=False);t=time.monotonic();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;rows=unpack(Path(a.encoded)/'held.json.gz');cpu=json.loads((Path(a.encoded)/'complete.json').read_text());assert cpu['status']=='PASS_COMPLETE_PURE_F0_NATIVE_GEOMETRY' and ref(Path(a.encoded)/'held.json.gz')['sha256']==cpu['files']['held']['compressed']['sha256'];assert ref(p['checkpoint']['path'])==p['checkpoint']
 assets={n:ref(Path(p['model'])/n) for n in p['model_files']};assert {n:r['sha256'] for n,r in assets.items()}==p['model_files'];enc,loading=AutoModel.from_pretrained(p['model'],local_files_only=True,attn_implementation='eager',add_pooling_layer=False,output_loading_info=True);save(O/'loading.json',loading);assert not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs'];assert all(k.startswith('cls.') or k in ['pooler.dense.weight','pooler.dense.bias'] for k in loading['unexpected_keys'])
 model=PatientSpanModel(enc);state=torch.load(p['checkpoint']['path'],map_location='cpu',weights_only=False);assert state['epoch']==24 and state['global_step']==768 and len(state['model'])==206;assert set(state['train_ids'])==set(cpu['train_ids']) and not set(state['train_ids'])&set(cpu['held_ids']);model.load_state_dict(state['model'],strict=True);assert all(torch.equal(v,state['model'][n]) for n,v in model.state_dict().items());save(O/'loaded.json',{'checkpoint':ref(p['checkpoint']['path']),'epoch':24,'step':768,'train_ids':state['train_ids'],'all206_cpu_values_exact':True,'model_source':ref(Path(__file__).with_name('model.py')),'no_fit':True});del state;model=model.cuda().eval();pred=[]
 for row in rows:
  start=time.monotonic()
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):records=model.infer(row,floor=-2.)
  x=packed(O/('raw-'+row['pmc_id']+'.json.gz'),{'pmc_id':row['pmc_id'],'patient_ids':[p['patient_id'] for p in row['patients']],'records':records,'all_candidate_count':len(row['candidates']),'global_logit_floor':-2.,'patient_logit_floor':None,'token_ids_sha256':row['token_ids_sha256']});x.update(pmc_id=row['pmc_id'],records=len(records),seconds=time.monotonic()-start);pred.append(x);save(O/'inference-progress.json',pred)
 assert len(pred)==16;save(O/'complete.json',{'status':'COMPLETE_FROZEN_PCL_F0_RAW','predictions':pred,'all_documents':16,'all_patients':47,'checkpoint':p['checkpoint'],'cpu':ref(Path(a.encoded)/'complete.json'),'seconds':time.monotonic()-t,'no_fit':True,'no_quality':True})
if __name__=='__main__':main()
