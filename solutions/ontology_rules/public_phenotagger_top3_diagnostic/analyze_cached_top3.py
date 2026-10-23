"""CPU-only literal-ID supply diagnostic on frozen top3; never emits predictions."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def checked_rows(spec):
    h=hashlib.sha256();n=0
    with Path(spec['path']).open('rb') as f:
        for line in f:
            h.update(line);n+=1;yield json.loads(line)
    assert h.hexdigest()==spec['sha256'] and Path(spec['path']).stat().st_size==spec['bytes']
    assert n==spec['rows'],(spec['path'],n)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args()
    start=time.monotonic();signal.alarm(115)
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    plan=json.loads(Path(a.plan).read_text());out=Path(plan['output_directory']);out.mkdir(exist_ok=False)
    assert sha(__file__)==plan['source_sha256']
    root=Path(plan['public_resource_root'])
    for rel,h in plan['resources'].items():assert sha(root/rel)==h,rel
    assert sha(plan['queries']['path'])==plan['queries']['sha256']
    qs=json.loads(Path(plan['queries']['path']).read_text())['queries'];assert len(qs)==298
    assert [q['query_index'] for q in qs]==list(range(298))
    docs={str(d['pmc_id']):d for d in map(json.loads,(root/'train64-cache/train64.input.jsonl').read_text().splitlines())}
    assert len(docs)==64 and all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs.values())
    query_spans=defaultdict(list)
    for q in qs:
        t=docs[q['pmc_id']]['full_text'][q['passage_index']]
        assert q['offset']==t['offset']+q['local_start'] and q['length']==q['local_end']-q['local_start']
        assert t['text'][q['local_start']:q['local_end']]==q['text']
        query_spans[q['pmc_id'],q['offset'],q['length']].append(q['query_index'])
    labels=(root/'lable.vocab').read_text().strip().splitlines()
    assert len(labels)==17266 and sha(plan['bioformer_label_file'])==sha(root/'lable.vocab')
    head=json.loads((root/'original-column-to-current-HPO.json').read_text())
    cmap={r['original_label']:r['current_label'] for r in head['mapping']}
    assert [r['original_label'] for r in head['mapping']]==labels
    dmap=json.loads((root/'native-pipeline/dictionary-current-HPO.json').read_text())
    for r in dmap['mapping']:
        if r['original_id'] in cmap:assert cmap[r['original_id']]==r['current_id']
        else:cmap[r['original_id']]=r['current_id']
    head_current={r['current_label'] for r in head['mapping'] if r['current_label']}
    items=[dict(query_index=i,model_ranks={},model_raw_rows={},dictionary_correct_id_positions=[],
                native_token_length=None,head_literal_present=q['gold_literal_id'] in set(labels),
                head_current_reachable=q['gold_literal_id'] in head_current,
                gold_same_span_atomic_count=len(q['gold_span_atomic_ids'])) for i,q in enumerate(qs)]
    models={};raw_stats={}
    for model,specs in plan['neural_raw'].items():
        began=time.monotonic();seen=set();n=0
        for sp in specs:
            for r in checked_rows(sp):
                n+=1;did=str(r['pmc_id']);assert did in docs;seen.add(did)
                key=(did,r['offset'],r['length'])
                if key not in query_spans:continue
                p=docs[did]['full_text'][r['passage_index']]
                assert r['offset']==p['offset']+r['local_start'] and r['length']==r['local_end']-r['local_start']
                assert p['text'][r['local_start']:r['local_end']]==r['text']
                cells=r['top3'];assert len(cells)==3
                for cell in cells:
                    assert labels[cell['column']]==cell['hpo_id']
                    assert math.isfinite(cell['probability']) and 0<=cell['probability']<=1
                assert all(cells[i]['probability']>=cells[i+1]['probability'] for i in range(2))
                for qi in query_spans[key]:
                    q=qs[qi];item=items[qi]
                    assert r['passage_index']==q['passage_index'] and r['text']==q['text']
                    item['model_raw_rows'][model]=item['model_raw_rows'].get(model,0)+1
                    nt=r['word_end_inclusive']-r['word_start']+1
                    assert item['native_token_length'] in (None,nt);item['native_token_length']=nt
                    ranks=[j+1 for j,c in enumerate(cells) if cmap.get(c['hpo_id'])==q['gold_literal_id']]
                    if ranks:item['model_ranks'][model]=min(item['model_ranks'].get(model,4),min(ranks))
        assert seen==set(docs) and n==296984
        raw_stats[model]=dict(rows=n,seconds=time.monotonic()-began)
        models[model]={str(k):{i for i,x in enumerate(items) if x['model_ranks'].get(model,4)<=k} for k in (1,2,3)}
    # Native positions come from the unchanged tokenizer/restore functions;
    # dictionary alternatives are checked only at matches actually produced by the native dictionary.
    sys.path.insert(0,str(root/'native-pipeline/src'))
    import nltk
    nltk.data.path=[str(root/'nltk_data')]
    from ssplit_tokenzier import ssplit_token_pos_lemma
    from restore_index import restore_index_nest_fn
    from dic_ner import dic_ont
    assert 'tensorflow' not in sys.modules and 'torch' not in sys.modules
    cached_dictionary=defaultdict(list)
    for sp in plan['dictionary_raw']:
        for r in checked_rows(sp):
            cached_dictionary[(str(r['pmc_id']),r['passage_index'])].append(
                (r['local_start'],r['local_end'],r['raw_hpo_id'],r['score']))
    dictionary=dic_ont({'dic_file':str(root/'native-pipeline/dict/noabb_lemma.dic'),
        'word_hpo_file':str(root/'native-pipeline/dict/word_id_map.json'),
        'hpo_word_file':str(root/'native-pipeline/dict/id_word_map.json')})
    qp=defaultdict(list)
    for q in qs:qp[q['pmc_id'],q['passage_index']].append(q)
    dictionary_seconds=time.monotonic()
    for (did,pi),queries in qp.items():
        text=docs[did]['full_text'][pi]['text'];conll=ssplit_token_pos_lemma(text)
        sentences=[[line.split('\t') for line in s.splitlines()] for s in conll.strip().split('\n\n')]
        annotation=''
        for sentence in sentences:
            words=[r[0] for r in sentence]
            annotation+=' '.join(words)+'\n'
            annotation+=''.join(str(j)+'\t'+str(j)+'\t'+word+'\tHP:None\t0.0\n' for j,word in enumerate(words))+'\n'
        locations=restore_index_nest_fn(text,annotation);at=0;locs=[]
        for sentence in sentences:
            locs.append([(int(x[0]),int(x[1])) for x in locations[at:at+len(sentence)]]);at+=len(sentence)
        assert at==len(locations)
        native=dictionary.matching(conll);restored=restore_index_nest_fn(text,native)
        actual=[(int(x[0]),int(x[1]),x[2],float(x[3])) for x in restored]
        assert actual==cached_dictionary[(did,pi)]
        dict_matches=defaultdict(list)
        for si,section in enumerate(native.strip().split('\n\n')):
            for line in section.splitlines()[1:]:
                lo,hi,term,first,_=line.split('\t');lo=int(lo);hi=int(hi)
                possible=dictionary.word_hpo.get(term,['None']);assert first==possible[0]
                span=(locs[si][lo][0],locs[si][hi][1]);dict_matches[span].append(possible)
        for q in queries:
            item=items[q['query_index']];native_lengths=[]
            for sent in locs:
                for lo,(s,_) in enumerate(sent):
                    if s!=q['local_start']:continue
                    for hi in range(lo,len(sent)):
                        if sent[hi][1]==q['local_end']:native_lengths.append(hi-lo+1)
            if native_lengths:
                assert len(set(native_lengths))==1
                assert item['native_token_length'] in (None,native_lengths[0]);item['native_token_length']=native_lengths[0]
            matches=dict_matches[(q['local_start'],q['local_end'])]
            item['native_dictionary_span_present']=bool(matches)
            item['native_dictionary_max_ids']=max(map(len,matches),default=0)
            item['dictionary_correct_id_positions']=sorted({j+1 for ids in matches for j,h in enumerate(ids) if cmap.get(h)==q['gold_literal_id']})
    dictionary_seconds=time.monotonic()-dictionary_seconds
    union={str(k):models['pubmed'][str(k)]|models['bioformer'][str(k)] for k in (1,2,3)}
    bothraw={i for i,x in enumerate(items) if x['model_raw_rows'].get('pubmed') and x['model_raw_rows'].get('bioformer')}
    assert {i for i,x in enumerate(items) if x['model_raw_rows'].get('pubmed')}==bothraw
    assert {i for i,x in enumerate(items) if x['model_raw_rows'].get('bioformer')}==bothraw
    dict_first={i for i,x in enumerate(items) if 1 in x['dictionary_correct_id_positions']}
    dict_any={i for i,x in enumerate(items) if x['dictionary_correct_id_positions']}
    no_raw=Counter()
    for i,x in enumerate(items):
        if i in bothraw:continue
        n=x['native_token_length']
        no_raw['native_unigram' if n==1 else 'unaligned_or_cross_sentence' if n is None else 'over_8_tokens' if n>8 else 'aligned_2_to_8_filtered']+=1
    def counts(sets):return {k:len(v) for k,v in sets.items()}
    result=dict(status='complete diagnostic only, no new candidates or predictions',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        plan_sha256=sha(a.plan),source_sha256=sha(__file__),queries_sha256=sha(plan['queries']['path']),
        denominator=298,unique_spans=len(query_spans),unique_gold_ids=len({q['gold_literal_id'] for q in qs}),
        head=dict(literal_present=sum(x['head_literal_present'] for x in items),mapped_current_reachable=sum(x['head_current_reachable'] for x in items),
            literal_absent=sum(not x['head_literal_present'] for x in items),mapped_unreachable=sum(not x['head_current_reachable'] for x in items),
            literal_present_but_mapped_away=sum(x['head_literal_present'] and not x['head_current_reachable'] for x in items)),
        same_span_multiple_gold_ids=sum(x['gold_same_span_atomic_count']>1 for x in items),
        raw_span_present_both_models=len(bothraw),raw_span_absent_reasons=no_raw,
        top_k_literal_support={m:counts(v) for m,v in models.items()},top_k_union_literal_support=counts(union),
        top2_or_3_increment_over_either_model_top1=len(union['3']-union['1']),
        top3_per_model_new_at_ranks_2_or_3={m:len(v['3']-v['1']) for m,v in models.items()},
        dictionary=dict(native_span_present=sum(x['native_dictionary_span_present'] for x in items),
            multi_id_span=sum(x['native_dictionary_max_ids']>1 for x in items),first_id_correct=len(dict_first),any_id_correct=len(dict_any),
            correct_only_after_first=len(dict_any-dict_first),after_first_and_no_neural_raw=len((dict_any-dict_first)-bothraw),
            after_first_native_unigram=sum(items[i]['native_token_length']==1 for i in dict_any-dict_first),
            later_ids_add_beyond_neural_top3=len((dict_any-dict_first)-union['3']),
            neural_top3_union_dictionary_all=len(union['3']|dict_any)),
        raw_reading=raw_stats,dictionary_passages=len(qp),dictionary_seconds=dictionary_seconds,seconds=time.monotonic()-start,
        no_new_NN=True,no_weights_loaded=True,originaldev_read=False,new_fit=False,
        limitations=plan['limitations'])
    audit=out/'per-query-audit.json';audit.write_text(json.dumps(items,indent=2)+'\n')
    result['audit']={'path':str(audit),'sha256':sha(audit),'bytes':audit.stat().st_size}
    assert result['seconds']<120
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
