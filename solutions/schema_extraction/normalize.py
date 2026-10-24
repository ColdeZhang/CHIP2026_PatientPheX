"""Apply the pre-frozen train32/public-only normalizer to selected relation tails."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--extraction',required=True);ap.add_argument('--normalizer-source',required=True);ap.add_argument('--references',required=True);ap.add_argument('--input',required=True);ap.add_argument('--train',required=True);ap.add_argument('--hpo',required=True);ap.add_argument('--bank',required=True);ap.add_argument('--model',required=True);ap.add_argument('--output',required=True);ap.add_argument('--device',default='cpu');args=ap.parse_args()
    refs=load(args.references);source=Path(args.normalizer_source)
    for name,item in refs['files'].items():assert sha(source/name)==item['sha256']
    sys.path.insert(0,str(source));from semantic import GroundedNormalizer
    start=time.monotonic();base=Path(args.extraction);complete=load(base/'complete.json');assert complete['mode']=='full' and complete['documents']==16
    assert sha(args.input)=='962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'
    docs={str(d['pmc_id']):d for d in [json.loads(x) for x in Path(args.input).read_text().splitlines() if x]}
    normalizer=GroundedNormalizer(args.hpo,args.train,model_dir=args.model,bank_dir=args.bank,policy_path=source/'policy_v1.json',device=args.device)
    normalizer.strict.assert_disjoint_input(args.input)
    files=sorted(base.glob('*.mapped.json'));assert len(files)==complete['windows']
    rows=[load(p) for p in files];assert set(r['pmc_id'] for r in rows)==set(docs)
    evidence=[];quotes=set()
    for row in rows:
        for relation in row['mapped']['relations']:
            head,tail=relation['head_entity'],relation['tail_entity']
            reason=(head['source_rejection'] or head['patient_rejection'] or tail['source_rejection'])
            if not reason and (head['type']!='patient' or tail['type']!='phenotype' or relation['type']!='has_phenotype'):
                reason='unexpected_relation_types'
            entry=dict(pmc_id=row['pmc_id'],window=row['window'],relation=relation,rejection=reason)
            if not reason:quotes.add(tail['source']['text'])
            evidence.append(entry)
    ordered=sorted(quotes);normalized=normalizer.normalize_many(ordered);lookup=dict(zip(ordered,normalized));pairs={pmc:{str(p['patient_id']):set() for p in doc['patient']} for pmc,doc in docs.items()}
    for entry in evidence:
        if entry['rejection']:continue
        head,tail=entry['relation']['head_entity'],entry['relation']['tail_entity'];norm=lookup[tail['source']['text']];entry['normalization']=norm
        if norm['status']=='abstain':entry['rejection']='normalization:'+norm['reason'];continue
        values=[tail['source']['text']] if norm['status']=='nil' else norm['ids']
        pairs[entry['pmc_id']][head['patient_id']].update(values)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    (out/'normalization.json').write_text(json.dumps(normalized,ensure_ascii=False,indent=2)+'\n')
    (out/'relations.json').write_text(json.dumps({pmc:{patient:sorted(v) for patient,v in ps.items()} for pmc,ps in pairs.items()},indent=2)+'\n')
    report=dict(documents=len(docs),patients=sum(len(ps) for ps in pairs.values()),selected_relations=len(evidence),unique_eligible_quotes=len(quotes),mapped_pairs=sum(len(v) for ps in pairs.values() for v in ps.values()),
                rejections=dict(Counter(x['rejection'] or 'accepted' for x in evidence)),normalizer_manifest=normalizer.strict.manifest,
                encoder_manifest=normalizer.encoder.manifest if normalizer.encoder else None,public_bank_reencoded=False,
                source_files=refs,extraction_complete_sha256=sha(base/'complete.json'),elapsed_seconds=time.monotonic()-start,
                outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir()})
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['documents','patients','selected_relations','unique_eligible_quotes','mapped_pairs','rejections','elapsed_seconds']}))
if __name__=='__main__':main()
