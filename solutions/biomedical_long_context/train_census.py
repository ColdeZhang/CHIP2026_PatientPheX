"""Train-only, published native-tokenizer geometry; no model/held labels.

Document passages are joined with two newlines. Original character offsets are
retained through explicit per-passage mapping; gaps are never invented as text.
ByteLevel lexical boundaries strip only exterior whitespace, never gold text.
"""
import argparse
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import tokenizers


def ref(p):
    p = Path(p); b = p.read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def chunks(n, width, overlap=128):
    assert width > overlap
    return [(a, min(a + width, n)) for a in range(0, n, width-overlap)
            if a == 0 or a + overlap < n]


def lexical(text, offsets):
    out=[]
    for a,b in offsets:
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b-1].isspace(): b -= 1
        out.append((a,b))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True)
    ap.add_argument('--tokenizer',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--hpo',required=True);ap.add_argument('--hpo-source',required=True)
    a=ap.parse_args();tr=[json.loads(x) for x in Path(a.train).read_text().splitlines()]
    assert len(tr)==32 and len({str(x['pmc_id']) for x in tr})==32
    # Reuse exactly the existing ontology admission function without importing
    # its unrelated sklearn model dependencies. No phenotype labels are added.
    tree=ast.parse(Path(a.hpo_source).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_hpo')
    ns={'Path':Path,'re':re};exec(compile(ast.Module(body=[fn],type_ignores=[]),a.hpo_source,'exec'),ns)
    _,allowed,aliases,hpo_version=ns['parse_hpo'](a.hpo)
    tok=tokenizers.Tokenizer.from_file(a.tokenizer);tok.no_padding();tok.no_truncation()
    totals=Counter(); per=[]; all_lengths=[]; entity_lengths=[]; passage_lengths=[]
    for d in tr:
        pmc=str(d['pmc_id']);parts=[];cursor=0;mapping=[]
        for pi,p in enumerate(d['full_text']):
            t=p['text']
            if parts:parts.append('\n\n');cursor+=2
            mapping.append((cursor,cursor+len(t),int(p['offset']),pi,t));parts.append(t);cursor+=len(t)
        text=''.join(parts);enc=tok.encode(text,add_special_tokens=False)
        assert len(enc.ids)==len(enc.offsets)
        offsets=lexical(text,enc.offsets)
        starts={x:i for i,(x,y) in enumerate(offsets) if y>x}
        ends={y:i for i,(x,y) in enumerate(offsets) if y>x}
        rawstarts={x:i for i,(x,y) in enumerate(enc.offsets) if y>x}
        rawends={y:i for i,(x,y) in enumerate(enc.offsets) if y>x}
        c=Counter();aligned=[];golds=set();all_lengths.append(len(enc.ids))
        for x in mapping:passage_lengths.append(len(tok.encode(x[4],add_special_tokens=False).ids))
        for e in d['entities']:
            if str(e.get('note','')).upper()=='NO':c['excluded_NO']+=1;continue
            eids={aliases.get(x.strip(),x.strip()) for x in str(e['identifier']).split(';')}
            if str(e['identifier'])!='-1' and not eids & allowed:
                c['excluded_nonphenotypic_ID']+=1;continue
            eo=int(e['offset']);eb=eo+int(e['length'])
            matches=[m for m in mapping if m[2]<=eo<eb<=m[2]+len(m[4])]
            assert len(matches)==1,(pmc,eo,len(matches))
            m=matches[0];x=m[0]+eo-m[2];y=x+int(e['length'])
            assert text[x:y]==e['text']
            key=(x,y)
            if key in golds:continue
            golds.add(key);c['positive_unique_spans']+=1
            if x in rawstarts and y in rawends:c['raw_exact_offset_aligned']+=1
            if x not in starts or y not in ends:c['lexical_offset_unaligned']+=1;continue
            i,j=starts[x],ends[y]
            assert i<=j and text[offsets[i][0]:offsets[j][1]]==e['text']
            assert eo==m[2]+offsets[i][0]-m[0]
            c['lexical_aligned']+=1;entity_lengths.append(j-i+1)
            ids=eids-{'-1',''}
            aligned.append({'i':i,'j':j,'pi':m[3],'surface':e['text'].casefold(),'ids':ids})
        for span in aligned:
            distant=[x for x in aligned if x is not span and x['pi']!=span['pi']
                     and (span['ids'] & x['ids'] or span['surface']==x['surface'])]
            if distant:c['with_same_concept_or_surface_other_passage']+=1
            # Co-visibility, not true evidence attribution or prediction recall.
            for width in (382,2046,4094,8190):
                ws=chunks(len(enc.ids),width,128)
                mine=[(s,t) for s,t in ws if s<=span['i']<=span['j']<t]
                if mine:c[f'positive_complete_in_{width+2}']+=1
                if any(s<=x['i']<=x['j']<t for s,t in mine for x in distant):
                    c[f'cross_passage_same_id_or_surface_covisible_{width+2}']+=1
        for width in (382,2046,4094,8190):
            ws=chunks(len(enc.ids),width,128)
            c[f'windows_{width+2}']=len(ws)
            c[f'encoded_tokens_{width+2}']=sum(t-s+2 for s,t in ws)
            c[f'full_square_cells_{width+2}']=sum((t-s+2)**2 for s,t in ws)
            c[f'band64_cells_{width+2}']=sum((t-s+2)*64 for s,t in ws)
        c['native_tokens_no_special']=len(enc.ids);c['passages']=len(mapping)
        totals.update(c);per.append({'pmc_id':pmc,**dict(c)})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    summary={'status':'TRAIN_ONLY_CPU_TOKENIZER_CENSUS','model_calls':0,'held_gold_reads':0,
      'sources':{'train':ref(a.train),'tokenizer':ref(a.tokenizer),
                 'hpo':ref(a.hpo),'hpo_admission_source':ref(a.hpo_source)},
      'hpo_version':hpo_version,'positive_admission':'exact old GP policy, NO excluded, NIL retained, any allowed phenotype descendant after alt_id canonicalization',
      'tokenizers_version':tokenizers.__version__,'normalizer':'published NFC',
      'mapping':'two-newline passage concatenation; published native ByteLevel offsets; trim exterior Unicode whitespace only',
      'counts':dict(totals),'documents':len(tr),'document_token_lengths':sorted(all_lengths),
      'passage_token_max':max(passage_lengths),'passages_gt382':sum(x>382 for x in passage_lengths),
      'documents_le8190':sum(x<=8190 for x in all_lengths),
      'positive_max_native_token_width':max(entity_lengths),
      'positive_native_width_histogram':dict(sorted(Counter(entity_lengths).items())),
      'limitations':['Train annotation geometry is not quality or error attribution.',
       'Same-ID/surface recurrence is not manual evidence; extra context may distract.',
       '384 census uses same new tokenizer and document sliding; old GP used passage windows.',
       'No deployment HF wrapper/attention/GPU tested; future native runtime must match all IDs.']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (out/'per-document.json').write_text(json.dumps(per,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
