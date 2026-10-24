#!/usr/bin/env python3
"""Fair padding/token-operation accounting from frozen input-only packing artifacts."""
import argparse,collections,datetime,gzip,json,time
from pathlib import Path
from census import chunks,load,ref,save


def costs(lengths):
    return dict(windows=len(lengths),actual_tokens=sum(lengths),
                actual_attention_cells=sum(n*n for n in lengths),
                fixed512_tokens=512*len(lengths),fixed512_attention_cells=512*512*len(lengths),
                dynamic4_tokens=sum(len(lengths[i:i+4])*max(lengths[i:i+4]) for i in range(0,len(lengths),4)),
                dynamic4_attention_cells=sum(len(lengths[i:i+4])*max(lengths[i:i+4])**2 for i in range(0,len(lengths),4)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory;start=time.monotonic()
    p=load(b/'budget-plan.json');assert ref(__file__)==p['source']
    for x in p['references'].values(): assert ref(x['path'])==x
    report=load(b/'packed-report.json')
    for x in report['artifacts'].values(): assert ref(x['path'])==x
    docs={x['pmc_id']:x for x in load(b/'packed-documents.json')}
    oldtotals=collections.defaultdict(collections.Counter);newtotals=collections.defaultdict(collections.Counter);per_doc=[]
    with gzip.open(b/'packed-token-map.jsonl.gz','rt') as f:
        for line in f:
            x=json.loads(line);d=docs[x['pmc_id']];split=d['split']
            plen=collections.Counter(t[0] for t in x['offset_map'] if t is not None)
            # Include empty passages: no window, but separator remains in the new stream.
            oldlengths=[r-l+2 for pi in sorted(plen) for l,r in chunks(plen[pi])]
            newlengths=[r-l+2 for l,r in chunks(len(x['input_ids']))]
            old,new=costs(oldlengths),costs(newlengths)
            assert old['windows']==d['prior_passage_windows'] and old['actual_tokens']==d['prior_passage_actual_tokens']
            assert new['windows']==d['windows'] and new['actual_tokens']==d['actual_encoded_tokens']
            oldtotals[split].update(old);newtotals[split].update(new)
            per_doc.append(dict(pmc_id=x['pmc_id'],split=split,passage=old,packed=new))
    identical=collections.defaultdict(dict)
    bydoc=collections.defaultdict(lambda:collections.defaultdict(list))
    for x in load(b/'packed-occurrences.json'):
        if x['kind']=='phenotype': bydoc[x['pmc_id']][x['key']].append((x['passage_index'],x['offset'],x['length']))
    groups=[]
    for pmc,concepts in bydoc.items():
        signatures=collections.defaultdict(list)
        for key,occ in concepts.items(): signatures[tuple(sorted(occ))].append(key)
        for signature,keys in signatures.items():
            if len(keys)>1: groups.append(dict(pmc_id=pmc,split=docs[pmc]['split'],concepts=sorted(keys),occurrences=[list(q) for q in signature]))
    results=dict(status='CPU_INPUT_ONLY_COST_ACCOUNTING',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start,
                 plan=ref(b/'budget-plan.json'),source=p['source'],passage=oldtotals,packed=newtotals,
                 ratios={s:{k:newtotals[s][k]/oldtotals[s][k] for k in oldtotals[s]} for s in oldtotals},
                 batch_policy='Stable within-document original window order, consecutive microbatches up to4. This is accounting, not actual model throughput. No length sorting or cross-document packing.',
                 duplicate_concept_occurrence_groups=groups,
                 query_indistinguishability='Distinct concept IDs with the same multiset of exact occurrences have identical mean endpoint embeddings. This head has no ID/name embeddings; no attempt to fix this by changing input.',
                 non_claims='Attention cells are per-head theoretical cells, exclude FFN/backward/activation lifetimes and padding-kernel behavior. Packed input has more real tokens despite fewer padded positions. Neither tensor bytes nor cell ratios are measured GPU memory or wall-time.',
                 per_document=per_doc,no_F1_gold=True,no_GPU=True)
    save(b/'budget-report.json',results)
    print(json.dumps({k:results[k] for k in ['seconds','passage','packed','ratios']}))
    print('identical-occurrence concept groups',len(groups))

if __name__=='__main__':main()
