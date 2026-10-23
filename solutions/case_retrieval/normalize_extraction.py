#!/usr/bin/env python3
"""Map exact LLM quotes to ontology candidates using training-only lexical retrieval."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from run import read_jsonl,load_hpo,LexicalMemory,alias_key,sha256,NEGATION

LINK_SYSTEM='''Map each quoted clinical phenotype to the best equivalent Human Phenotype Ontology candidate.
Return a JSON object from item codes to candidate HPO identifiers, or "-1" if none is an equivalent phenotype. Choose only a candidate listed for that item. Preserve meaning, specificity, anatomy and direction (increased vs decreased). A related disease or anatomy alone is not equivalent. For a phrase expressing multiple independent findings, choose semicolon-separated candidates only if every selected phenotype is explicitly expressed. Do not infer unmentioned symptoms. No explanation or markdown.'''


def parse_json(text):
    text=text.strip()
    if text.startswith('<think>') and '</think>' in text:text=text.split('</think>',1)[1].strip()
    text=re.sub(r'^```(?:json)?\s*','',text);text=re.sub(r'\s*```$','',text)
    try:return json.loads(text)
    except json.JSONDecodeError:
        for opener,closer in [('[',']'),('{','}')]:
            a=text.find(opener);b=text.rfind(closer)
            if 0<=a<b:
                try:return json.loads(text[a:b+1])
                except json.JSONDecodeError:pass
        return None


def occurrences(prompts,responses):
    lookup={x['row_id']:x for x in prompts};found=defaultdict(dict);stats=Counter()
    for response in responses:
        key=response['row_id'];prompt=lookup[key];data=parse_json(response['response'])
        if isinstance(data,dict):data=data.get('mentions',data.get('phenotypes'))
        if not isinstance(data,list):stats['invalid_json']+=1;continue
        stats['valid_responses']+=1
        for item in data:
            quote=item if isinstance(item,str) else item.get('text','') if isinstance(item,dict) else ''
            if not isinstance(quote,str) or len(quote.strip())<3:stats['invalid_quote']+=1;continue
            matched=False
            for segment in prompt['segments']:
                for m in re.finditer(re.escape(quote),segment['text']):
                    matched=True;before=re.split(r'[.!?;]',segment['text'][max(0,m.start()-100):m.start()])[-1]
                    if NEGATION.search(before) and not re.search(r'\b(?:but|however|although|not only)\b',before,re.I):
                        stats['negated_occurrence']+=1;continue
                    offset=segment['offset']+m.start();context=segment['text'][max(0,m.start()-100):m.end()+100]
                    found[prompt['pmc_id']][(offset,len(quote),quote)]={'text':quote,'offset':offset,'length':len(quote),'context':context}
            if not matched:stats['unmatched_quote']+=1
    return {pmc:sorted(items.values(),key=lambda e:(e['offset'],e['length'])) for pmc,items in found.items()},dict(stats)


class Normalizer:
    def __init__(self,train,hpo):
        terms,valid,canonical,version=load_hpo(hpo)
        self.terms=terms;self.lexical=LexicalMemory(terms,valid,canonical,train)
        self.keys=sorted(self.lexical.aliases);self.surfaces=[' '.join(x) for x in self.keys]
        self.vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,sublinear_tf=True,dtype=np.float32)
        self.matrix=self.vec.fit_transform(self.surfaces)
    def candidates(self,quote,context,k=8):
        key=alias_key(quote);exact=self.lexical.aliases.get(key)
        if exact:
            selected=self.lexical.choose_id(key,tuple(sorted(exact)),context)
            return [{'identifier':selected,'alias':' '.join(key),'name':'; '.join(self.terms.get(i,{}).get('name',quote) for i in selected.split(';')),'score':1.0}],True
        scores=(self.vec.transform([' '.join(key)])@self.matrix.T).toarray()[0];result=[];seen=set()
        for ix in np.argsort(scores)[::-1][:120]:
            if scores[ix]<=0:break
            for identifier in sorted(self.lexical.aliases[self.keys[ix]]):
                if identifier in seen:continue
                seen.add(identifier);result.append({'identifier':identifier,'alias':self.surfaces[ix],'name':'; '.join(self.terms.get(i,{}).get('name',quote) for i in identifier.split(';')),'score':float(scores[ix])})
                if len(result)>=k:return result,False
        return result,False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',required=True);ap.add_argument('--hpo',required=True);ap.add_argument('--input',required=True)
    ap.add_argument('--prompts',required=True);ap.add_argument('--responses',required=True);ap.add_argument('--output-dir',required=True)
    ap.add_argument('--candidate-count',type=int,default=8);ap.add_argument('--items-per-prompt',type=int,default=18)
    ap.add_argument('--dense-candidates',help='Optional frozen SapBERT candidate JSON for these exact quote item codes')
    args=ap.parse_args();train=read_jsonl(args.train);inputs=read_jsonl(args.input)
    assert not({str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in inputs})
    assert not any(d.get('entities') or d.get('association') for d in inputs)
    prompts=[json.loads(x) for x in Path(args.prompts).read_text().splitlines() if x.strip()];responses=[json.loads(x) for x in Path(args.responses).read_text().splitlines() if x.strip()]
    assert len({x['row_id'] for x in responses})==len(responses)
    found,stats=occurrences(prompts,responses);norm=Normalizer(train,args.hpo);items=[];lookup={};exact_count=0
    for doc in inputs:
        for occurrence in found.get(str(doc['pmc_id']),[]):
            quote=occurrence['text'];key=quote.casefold()+'\n'+occurrence['context']
            if key not in lookup:
                candidates,exact=norm.candidates(quote,occurrence['context'],args.candidate_count)
                code=f'M{len(lookup):05d}';lookup[key]=code
                items.append({'code':code,'quote':quote,'context':occurrence['context'],'candidates':candidates,'exact':exact,'identifier':candidates[0]['identifier'] if candidates else '-1'})
                exact_count+=int(exact)
            occurrence['item_code']=lookup[key]
    if args.dense_candidates:
        dense={x['code']:x for x in json.loads(Path(args.dense_candidates).read_text())}
        for item in items:
            if item['exact'] or item['code'] not in dense:continue
            extra=dense[item['code']];assert extra['quote']==item['quote'],'Dense candidate quote/code mismatch'
            seen={x['identifier'] for x in item['candidates']}
            for c in extra['candidates']:
                if c['identifier'] in seen:continue
                if any(i not in norm.terms for i in c['identifier'].split(';')):continue
                c=dict(c);c['name']='; '.join(norm.terms[i]['name'] for i in c['identifier'].split(';'));item['candidates'].append(c);seen.add(c['identifier'])
    pending=[x for x in items if not x['exact']];link_prompts=[]
    for start in range(0,len(pending),args.items_per_prompt):
        batch=pending[start:start+args.items_per_prompt];lines=[]
        for x in batch:
            lines.append(f"{x['code']}: quote={json.dumps(x['quote'])}; context={json.dumps(x['context'])}\nCandidates: "+json.dumps([{'id':c['identifier'],'name':c['name'],'alias':c['alias']} for c in x['candidates']],ensure_ascii=False))
        link_prompts.append({'row_id':f'link:{start}','pmc_id':'normalization','item_codes':[x['code'] for x in batch],'messages':[{'role':'system','content':LINK_SYSTEM},{'role':'user','content':'\n\n'.join(lines)}]})
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'normalization-items.json').write_text(json.dumps({'occurrences':found,'items':items},ensure_ascii=False))
    (out/'semantic-queries.json').write_text(json.dumps([{'code':x['code'],'quote':x['quote']} for x in pending],ensure_ascii=False))
    (out/'alias-records.json').write_text(json.dumps([{'text':text,'identifiers':sorted(norm.lexical.aliases[key])} for key,text in zip(norm.keys,norm.surfaces)],ensure_ascii=False))
    (out/'link-prompts.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in link_prompts))
    manifest={'configuration':vars(args),'parse':stats,'occurrences':sum(map(len,found.values())),'distinct_items':len(items),'exact_items':exact_count,'link_prompts':len(link_prompts),'files':{k:sha256(getattr(args,k)) for k in ['train','hpo','input','prompts','responses']+(['dense_candidates'] if args.dense_candidates else [])},'code_sha256':sha256(__file__)}
    (out/'normalization-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))

if __name__=='__main__':main()
