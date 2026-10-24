"""CPU data/target API contract only: no model load, forward, loss, or fitting."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-assets',required=True);ap.add_argument('--train',required=True);ap.add_argument('--output',required=True);args=ap.parse_args();start=time.monotonic()
    import torch
    from gliner2.models.base import load_extractor_tokenizer
    from gliner2.processor import SchemaTransformer
    from gliner2.training import ExtractorTrainer,TrainingConfig,InputExample,Relation
    assert not torch.cuda.is_available(),'CPU contract must hide CUDA'
    assert hashlib.sha256(Path(args.train).read_bytes()).hexdigest()=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    processor=SchemaTransformer(tokenizer=load_extractor_tokenizer(args.model_assets))
    text='Ann has pain. Ann has fever.'
    span=lambda a,b:dict(text=text[a:b],start=a,end=b)
    example=InputExample(text=text,entities={'patient':[span(0,3),span(14,17)],'phenotype':[span(8,12),span(22,27)]},relations=[Relation('has_phenotype',head=span(14,17),tail=span(22,27))])
    assert example.validate()==[]
    record=example.to_dict();assert InputExample.from_dict(record).to_dict()==record
    def collate(record):
        return processor.collate_fn_train([(record['input'],record['output'])],max_len=None,error_policy='raise',architecture='boundary',max_gold_per_query=64,on_capacity_exceeded='raise',dedupe_contained_entity_spans=False)
    batch=collate(record);pairs,mask,*rest=batch.targets.edge_targets;actual=pairs[mask].tolist();assert actual==[[4,5,6,7]],actual
    # Actual positive entity annotations alone are valid supervision. We do not
    # attach document-level relations to arbitrary mention pairs in this sample.
    docs=[json.loads(x) for x in Path(args.train).read_text().splitlines() if x];real=None
    for doc in docs:
        for index,p in enumerate(doc['full_text']):
            tokens=list(processor.word_splitter(p['text'],lower=False));starts={a for t,a,b in tokens};ends={b for t,a,b in tokens};spans=[]
            for e in doc['entities']:
                if str(e.get('note') or '').upper()=='NO':continue
                a=e['offset']-p['offset'];b=a+e['length']
                if a in starts and b in ends and p['text'][a:b]==e['text']:
                    spans.append(dict(text=e['text'],start=a,end=b))
            if spans:
                spans=list({(s['start'],s['end']):s for s in spans}.values())
                if len(spans)>64:continue
                ex=InputExample(text=p['text'],entities={'phenotype':spans});assert ex.validate()==[];out=collate(ex.to_dict());assert int(out.targets.mention_mask.sum())==len(spans)
                real=dict(pmc_id=str(doc['pmc_id']),passage_index=index,positive_spans=len(spans),target_mentions=int(out.targets.mention_mask.sum()),input_ids=int(out.input_ids.shape[-1]),relation_labels=0);break
        if real:break
    assert real is not None
    report=dict(status='CPU_NATIVE_BOUNDARY_DATA_TARGET_CONTRACT_PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),upstream_entry='gliner2.training.ExtractorTrainer.train',training_signature=str(inspect.signature(ExtractorTrainer.train)),config_supports=dict(dedupe_contained_entity_spans='dedupe_contained_entity_spans' in TrainingConfig.__dataclass_fields__,on_capacity_exceeded='on_capacity_exceeded' in TrainingConfig.__dataclass_fields__),synthetic_same_name_unique_coordinate_edge=actual,actual_train32_entity_only_sample=real,model_parameters_loaded=0,forward_calls=0,backward_calls=0,optimizer_steps=0,quality_evaluations=0,elapsed_seconds=time.monotonic()-start)
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
