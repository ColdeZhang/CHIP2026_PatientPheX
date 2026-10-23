#!/usr/bin/env python3
"""Prepare label-free whole-document ontology evidence for a local LLM.

Unlike retrieved demonstrations, this route is zero-shot: supplied patient
markers, ontology names and textual spans are the only task-specific context.
"""
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import run as baseline

SYSTEM='''You are a clinical evidence annotator for biomedical case reports. Identify which candidate phenotypes are actually present in each specified individual patient.
Return exactly one JSON object mapping EVERY supplied patient ID to a list of candidate codes, with empty lists for patients without supporting evidence. Use only candidate codes present in this evidence window. No prose or markdown.
The inline [PATIENT id: original words] markers are authoritative patient identities at those spans. Resolve pronouns and case/family discourse using these markers. A phenotype may belong to multiple specified patients when the text explicitly supports this.
A candidate list is a list of possible text concepts, not assertions that patients have them. Exclude explicit negatives, normal findings, unaffected relatives, hypothetical risks, prior unrelated cases and general disease descriptions. Do not infer symptoms merely from a diagnosis. A section heading such as INTRO or DISCUSSION may still contain this patient's clinical evidence; decide from the content.
Keep a historical phenotype when that patient actually had it, even if it later resolved or was treated. Family members count only if they are among the supplied patient IDs and have evidence. Do not merge different individuals. Every selected code needs support for that specific patient in the supplied window; use the broader context for coreference, not disease-based guessing.'''


def mark(text,offset,doc):
    intervals=defaultdict(set)
    for p in doc.get('patient',[]):
        for m in p.get('mention',[]):
            a=int(m.get('offset',-1))-offset;b=a+int(m.get('length',len(m.get('text',''))))
            if 0<=a<b<=len(text):intervals[(a,b)].add(str(p['patient_id']))
    out=[];pos=0
    for (a,b),pids in sorted(intervals.items()):
        if a<pos:continue
        out.extend([text[pos:a],f"[PATIENT {'/'.join(sorted(pids))}: {text[a:b]}]"]);pos=b
    out.append(text[pos:]);return ''.join(out)


def pieces(doc,max_chars):
    out=[]
    for pi,p in enumerate(doc.get('full_text',[])):
        if str(p.get('section_type','')).upper() in baseline.STOP_SECTIONS:continue
        text=p.get('text','');offset=int(p.get('offset',0))
        start=0
        while start<len(text):
            end=min(len(text),start+max_chars)
            if end<len(text):
                boundary=max(text.rfind('. ',start+max_chars//2,end),text.rfind('; ',start+max_chars//2,end))
                if boundary>=0:end=boundary+1
            out.append({'offset':offset+start,'end':offset+end,'section':str(p.get('section_type','')),'passage':pi,'text':text[start:end]})
            start=end
    return out


def windows(parts,budget,overlap=1):
    result=[];current=[];size=0
    for part in parts:
        n=len(part['text'])+80
        if current and size+n>budget:
            result.append(current);current=current[-overlap:] if overlap else [];size=sum(len(p['text'])+80 for p in current)
        current.append(part);size+=n
    if current:result.append(current)
    return result


def build(doc,pred,terms,max_chars,patient_codes=False):
    id_map={f'P{i:03d}':str(p['patient_id']) for i,p in enumerate(doc.get('patient',[]),1)} if patient_codes else None
    if id_map:
        inverse={v:k for k,v in id_map.items()}
        doc=dict(doc,patient=[dict(p,patient_id=inverse[str(p['patient_id'])]) for p in doc['patient']])
    groups=defaultdict(list)
    for e in pred['entities']:
        for hid in ([e['text']] if e['identifier']=='-1' else e['identifier'].split(';')):groups[hid].append(e)
    codes={hid:f'C{i:03d}' for i,hid in enumerate(sorted(groups),1)}
    patients=[str(p['patient_id']) for p in doc.get('patient',[])]
    # Aliases are supplied metadata, never inferred from labels or global ID meaning.
    patients_text='\n'.join(str(p['patient_id'])+': '+ '; '.join(sorted(set(m.get('text','') for m in p.get('mention',[])))[:12]) for p in doc.get('patient',[]))
    result=[]
    for wi,parts in enumerate(windows(pieces(doc,min(5500,max_chars//2)),max_chars)):
        visible={hid:[e for e in es if any(p['offset']<=e['offset'] and e['offset']+e['length']<=p['end'] for p in parts)] for hid,es in groups.items()}
        visible={hid:es for hid,es in visible.items() if es}
        if not visible:continue
        cmap={codes[hid]:hid for hid in sorted(visible)}
        lines=[]
        for hid,es in visible.items():
            name=terms.get(hid,{}).get('name','unlinked phenotype text')
            surfaces=' / '.join(sorted({e['text'] for e in es})[:6])
            lines.append(f'{codes[hid]} | {hid} | {name} | observed words: {surfaces}')
        evidence='\n\n'.join(f"Passage {p['passage']} [{p['section']}], original offset {p['offset']}:\n"+mark(p['text'],p['offset'],doc) for p in parts)
        content='Supplied patient identities:\n'+patients_text+'\n\nCandidate ontology concepts in this window:\n'+'\n'.join(sorted(lines))+'\n\nFull evidence in document order:\n'+evidence
        result.append({'row_id':str(doc['pmc_id'])+':'+str(wi),'pmc_id':doc['pmc_id'],'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':content}],'patient_ids':patients,'code_map':cmap,'evidence_ranges':[[p['offset'],p['end']] for p in parts],'candidate_evidence':{codes[hid]:[{'offset':e['offset'],'length':e['length'],'text':e['text']} for e in es] for hid,es in visible.items()},'training_demonstrations':[]})
        if id_map:result[-1]['patient_id_map']=id_map
    return result


def main():
    ap=argparse.ArgumentParser()
    for k in ['input','pred','hpo','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--max-context-chars',type=int,default=22000)
    ap.add_argument('--patient-codes',action='store_true',help='Use reversible P001 identity keys to avoid delimiter ambiguity in patient IDs')
    a=ap.parse_args();inputs=baseline.read_jsonl(a.input);preds={d['pmc_id']:d for d in baseline.read_jsonl(a.pred)}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    assert {d['pmc_id'] for d in inputs}==set(preds)
    terms,allowed,alt,version=baseline.load_hpo(a.hpo)
    rows=[r for d in inputs for r in build(d,preds[d['pmc_id']],terms,a.max_context_chars,a.patient_codes)]
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    meta={'method':'zero-shot-ontology-evidence-patient-coreference','config':vars(a),'source_sha256':baseline.sha256(__file__),'inputs':{k:baseline.sha256(getattr(a,k)) for k in ['input','pred','hpo']},'prompt_sha256':baseline.sha256(output),'documents':len(inputs),'rows':len(rows),'max_prompt_chars':max([sum(len(m['content']) for m in r['messages']) for r in rows] or [0]),'training_demonstrations':False,'input_annotations_used':False}
    output.with_suffix('.manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
