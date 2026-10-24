"""Six-shape actual-schema CPU check; reuse the completed320-target audit."""
import argparse
import dataclasses
import json
from pathlib import Path
import random
import time
from probe import observe_batch, write, sha


def main():
    ap=argparse.ArgumentParser()
    for n in ('model','train','audit','plan','output'):ap.add_argument('--'+n,required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    import torch
    from tokenizers import Tokenizer
    from torch.utils.data import DataLoader
    from gliner2.models.base import load_extractor_tokenizer
    from gliner2.processor import SchemaTransformer
    from gliner2.training.trainer import ExtractorDataset,ExtractorCollator
    from gliner2.training.sampler import LengthGroupedSampler
    assert not torch.cuda.is_available();torch.set_num_threads(1);random.seed(20261024);torch.manual_seed(20261024)
    plan=json.loads(Path(args.plan).read_text());assert sha(args.train)==plan['data']['train_native_sha256'];assert sha(args.audit)==plan['data']['target_audit_sha256']
    rows=[json.loads(x) for x in Path(args.train).read_text().splitlines()];audits=[json.loads(x) for x in Path(args.audit).read_text().splitlines()];indices=plan['probe']['indices'];selected=[rows[i] for i in indices];lookup={}
    for i in indices:
        lookup[rows[i]['input']]=i;lookup[rows[i]['input']+'.']=i
    processor=SchemaTransformer(tokenizer=load_extractor_tokenizer(args.model));assert processor.sampling_config.synthetic_entity_label_prob==.2
    collate=ExtractorCollator(processor,is_training=True,max_len=4096,architecture='boundary',max_gold_per_query=64,on_capacity_exceeded='raise',allow_invalid_samples=False,dedupe_contained_entity_spans=False)
    ds=ExtractorDataset(selected,max_samples=-1,shuffle=True,seed=20261024,validate=True);sampler=LengthGroupedSampler(ds.lengths,2,window_batches=50,seed=20261024)
    loader=DataLoader(ds,batch_size=2,sampler=sampler,num_workers=0,drop_last=True,collate_fn=lambda raw:raw)
    native=Tokenizer.from_file(str(Path(args.model)/'tokenizer.json'));cache={};reports=[]
    for epoch in range(2):
        sampler.set_epoch(epoch);seen=[]
        for batch_id,raw in enumerate(loader):
            rng_before=random.getstate();batch=collate(raw);rng_after=random.getstate()
            observed=observe_batch(batch,lookup,audits,processor,native,cache)
            write(out/f'epoch{epoch+1}-batch{batch_id}.json',{'rows':observed,'rng_before':rng_before,'rng_after':rng_after})
            for r in observed:
                assert r['actual_input_ids']==r['independent_tokenizer_json_ids'];assert sorted(r['actual_positive_pairs'])==sorted(r['expected_positive_pairs']);assert r['query_mask']==[True] and r['input_ids']<=4096;seen.append(r['row_id'])
            reports.extend({'epoch':epoch+1,'row_id':r['row_id'],'ids':r['input_ids'],'schema_tokens':r['actual_schema_tokens'],'gold':len(r['actual_positive_pairs'])} for r in observed)
        assert sorted(seen)==sorted(indices)
    result={'status':'SIX_SHAPES_NATIVE_STOCHASTIC_SCHEMA_CURRENT_IDS_AND_TARGETS_PASS_NO_MODEL','sampling_config':dataclasses.asdict(processor.sampling_config),'epochs':2,'batch_size':2,'batches':6,'observations':reports,'max_actual_ids':max(r['ids'] for r in reports),'max_gold':max(r['gold'] for r in reports),'native_query_augmentation_unchanged':True,'model_parameters_loaded':0,'forward_calls':0,'optimizer_updates':0,'elapsed_seconds':time.monotonic()-start,'source_sha256':sha(__file__),'shared_observer_sha256':sha(Path(__file__).with_name('probe.py'))}
    write(out/'complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
