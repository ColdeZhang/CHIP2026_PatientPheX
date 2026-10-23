#!/usr/bin/env python3
"""CPU readiness for the fixed native BioMed train32 control; no validation gold."""
import argparse, ast, hashlib, importlib.metadata, json, time
from pathlib import Path
import check_gliner_biomed
import train_gliner_biomed as trainer
from run_gliner_pilot import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model-dir','tokenizer-dir','train','excluded-input','data-manifest','output','parent-trainer','expected-weight-sha']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    started=time.monotonic()
    source=Path(trainer.__file__).read_text()
    parent=Path(a.parent_trainer).read_text()
    expected=parent.replace('import run_gliner_pilot\nfrom run_gliner_pilot import load, sha','import check_gliner_biomed\nfrom check_gliner_biomed import load\nfrom run_gliner_pilot import sha').replace("'loader_source_sha256': sha(run_gliner_pilot.__file__)","'loader_source_sha256': sha(check_gliner_biomed.__file__)")
    assert source==expected, 'Only native loader and its truthful fingerprint may differ'
    meta=json.loads(Path(a.data_manifest).read_text());rows=[json.loads(x) for x in Path(a.train).read_text().splitlines()];inputs=[json.loads(x) for x in Path(a.excluded_input).read_text().splitlines()]
    assert sha(a.train)==meta['artifacts']['nested_train32']['sha256']
    ids={str(x['pmc_id']) for x in rows};excluded={str(x['pmc_id']) for x in inputs}
    assert len(rows)==1150 and len(ids)==32 and len(excluded)==16 and not ids&excluded
    assert ids==set(meta['artifacts']['nested_train32']['pmc_ids']) and excluded==set(meta['artifacts']['nested_validation16']['pmc_ids'])
    assert all('entities' not in x and 'association' not in x for x in inputs)
    assert sha(Path(a.model_dir)/'pytorch_model.bin')==a.expected_weight_sha
    model=check_gliner_biomed.load(Path(a.model_dir),Path(a.tokenizer_dir))
    n=sum(x.numel() for x in model.parameters());assert n==477981696
    counts={'rows':len(rows),'documents':len(ids),'positive_instances':0,'valid_spans':0,'max_tokens':0}
    for i in range(0,len(rows),4):
        batch=trainer.collate(model.data_processor,rows[i:i+4])
        counts['positive_instances']+=int((batch['labels']>0).sum());counts['valid_spans']+=int(batch['span_mask'].sum());counts['max_tokens']=max(counts['max_tokens'],int(batch['attention_mask'].sum(-1).max()))
    scores=torch.tensor([[[2.],[-2.],[999.]]],requires_grad=True);labels=torch.tensor([[[1.],[0.],[1.]]]);mask=torch.tensor([[1,1,0]])
    loss=trainer.masked_span_bce(scores,labels,mask);loss.backward();g=scores.grad.flatten().tolist()
    assert abs(float(loss)-.126928011)<1e-6 and g[0]<0<g[1] and g[2]==0
    longest=sorted(rows,key=lambda x:(-x['tokens'],-len(x['tokenized_text']),x['row_id']))[:4]
    assert not torch.cuda.is_initialized()
    result={'scope':'CPU train32 readiness only','configuration':vars(a),'parameters':n,'strict_load':True,'counts':counts,'all_1150_rows_actual_collate_pass':True,'parent_trainer_only_loader_and_fingerprint_delta':True,'loss_probe':{'value':float(loss.detach()),'gradients':g,'padding_zero':True},'longest_batch_rows':[x['row_id'] for x in longest],'longest_batch_tokens':[x['tokens'] for x in longest],'seconds':time.monotonic()-started,'cuda_initialized':False,'sources':{'checker':sha(__file__),'trainer':sha(trainer.__file__),'native_loader':sha(check_gliner_biomed.__file__),'parent_trainer':sha(a.parent_trainer)},'inputs':{k:sha(getattr(a,k)) for k in ['train','excluded_input','data_manifest']},'actual_weight_sha256':a.expected_weight_sha,'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','gliner']}}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['parameters','counts','seconds','cuda_initialized','loss_probe']}),flush=True)

if __name__=='__main__':main()
