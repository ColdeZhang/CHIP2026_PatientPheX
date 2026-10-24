"""Frozen train32/public41099 normalization of legal native entity quotes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from evidence_io import write,sha
from admission_v3 import admit


def main():
    ap=argparse.ArgumentParser()
    for n in ('extraction','normalizer-source','references','input','train','hpo','bank','model','output'):ap.add_argument('--'+n,required=True)
    ap.add_argument('--device',default='cuda');args=ap.parse_args();admit('normalizer_before_source_import');started=time.monotonic();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    refs=json.loads(Path(args.references).read_text());source=Path(args.normalizer_source)
    for n,r in refs['files'].items():assert sha(source/n)==r['sha256']
    sys.path.insert(0,str(source));from semantic import GroundedNormalizer;from strict import verify_span
    assert sha(args.input)=='ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37';assert sha(args.train)=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    docs={str(d['pmc_id']):d for d in [json.loads(x) for x in Path(args.input).read_text().splitlines()]};assert len(docs)==16 and sum(len(d['patient']) for d in docs.values())==47
    root=Path(args.extraction);complete=json.loads((root/'complete.json').read_text());assert complete['status']=='COMPLETE_F0_NATIVE_TASK1_EXTRACTION' and complete['windows']==172
    assert sha(root/'selected.json')==complete['selected_sha256'];selected=json.loads((root/'selected.json').read_text());assert len(selected)==complete['selected_windows_entities'];spans={};rejected=Counter()
    for row in selected:
        if row['rejection'] is not None:rejected[row['rejection']]+=1;continue
        s=row['source'];verify_span(docs[row['pmc_id']],s['text'],s['offset']);key=(row['pmc_id'],s['offset'],s['length'],s['text'])
        if key not in spans or row['confidence']>spans[key]['confidence']:spans[key]=row
    normalizer=GroundedNormalizer(args.hpo,args.train,model_dir=args.model,bank_dir=args.bank,policy_path=source/'policy_v1.json',device=args.device);normalizer.strict.assert_disjoint_input(args.input)
    quotes=sorted({key[3] for key in spans});strict=[normalizer.strict.normalize(q) for q in quotes]
    from semantic import precheck
    pending=sorted({x['quote'] for x in strict if x['status']=='abstain' and x['reason']==normalizer.policy['semantic_fallback_reason'] and not precheck(x['quote'],normalizer.policy)})
    write(out/'query-plan.json',{'quotes':quotes,'semantic_queries':pending,'unique_legal_spans':len(spans),'selected_count':len(selected),'normalizer_manifest':normalizer.strict.manifest,'frozen_policy_sha256':normalizer.policy_sha256})
    admit('normalizer_immediately_before_public_encoder_load');normalized=normalizer.normalize_many(quotes);write(out/'normalization.json',normalized);lookup=dict(zip(quotes,normalized));entities={pmc:[] for pmc in docs};accepted=[];atoms=set()
    for (pmc,offset,length,text),row in sorted(spans.items()):
        norm=lookup[text];entry={'pmc_id':pmc,'source':row['source'],'confidence':row['confidence'],'normalization':norm};accepted.append(entry)
        if norm['status']=='abstain':continue
        assert norm['status'] in ('mapped','nil');ids=norm['ids'];assert ids==['-1'] if norm['status']=='nil' else '-1' not in ids
        for hid in ids:
            key=(pmc,offset,length,hid,None)
            if key in atoms:continue
            atoms.add(key);entities[pmc].append({'identifier':hid,'offset':offset,'length':length,'text':text,'type':'Phenotype','note':None})
    write(out/'evidence.json',accepted);write(out/'entities.json',entities)
    report={'status':'COMPLETE_FIXED_TASK1_NORMALIZATION','documents':16,'patients':47,'legal_unique_spans':len(spans),'unique_quotes':len(quotes),'semantic_queries':len(pending),'normalization_status':dict(Counter(x['status'] for x in normalized)),'accepted_entity_atoms':sum(len(v) for v in entities.values()),'normalizer_manifest':normalizer.strict.manifest,'encoder_manifest':normalizer.encoder.manifest if normalizer.encoder else None,'public_bank_reencoded':False,'policy_sha256':normalizer.policy_sha256,'no_additional_overlap_conflict_NO_NIL_filter':True,'source_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-started,'entities_sha256':sha(out/'entities.json'),'source_refs':refs}
    write(out/'complete.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['source_refs','normalizer_manifest','encoder_manifest']}),flush=True)


if __name__=='__main__':main()
