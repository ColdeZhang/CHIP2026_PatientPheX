#!/usr/bin/env python3
"""Prepare document-level clinical evidence and train-only retrieved demonstrations."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from run import read_jsonl, sha256


SYSTEM = """You identify the clinical phenotypes of specific patients in biomedical case reports.
Return exactly one JSON object mapping EVERY supplied patient ID to a list of candidate codes.
Use only the listed candidate codes. A code means that its phenotype is present in that patient.
Use the patient's aliases and the full clinical evidence, including later paragraphs and coreference.
Do not assign a phenotype from general disease descriptions, references, other patients, relatives,
or an explicitly negative finding. Do not infer unmentioned symptoms from a disease name.
If a patient has no supported phenotype, return an empty list. No explanation or markdown."""


def concepts_of(entities):
    groups = defaultdict(list)
    for entity in entities:
        if str(entity.get("note", "")).upper() == "NO":
            continue
        ids = [entity["text"]] if str(entity["identifier"]) == "-1" else str(entity["identifier"]).split(";")
        for identifier in ids:
            groups[identifier].append(entity)
    return dict(groups)


def patient_text(doc):
    return "\n".join(str(p["patient_id"]) + ": " + "; ".join(sorted(set(m["text"] for m in p.get("mention", []))))
                     for p in doc.get("patient", []))


def context_text(doc, entities, char_budget):
    selected = []
    mentions = [m for p in doc.get("patient", []) for m in p.get("mention", [])]
    for index, passage in enumerate(doc["full_text"]):
        begin = int(passage["offset"]); end = begin+len(passage["text"])
        patient_mentions = [m for m in mentions if begin <= int(m["offset"]) < end]
        phenotype_mentions = [e for e in entities if begin <= int(e["offset"]) < end]
        section = str(passage.get("section_type", "")) + " " + str(passage.get("type", ""))
        if re.search(r"ref|acknow|author|comp_int", section, re.I):
            continue
        clinical = bool(re.search(r"case|patient|clinical|result", section, re.I))
        priority = 10*bool(patient_mentions) + 3*clinical + min(5, len(phenotype_mentions))
        if priority == 0:
            continue
        body = passage["text"]
        if len(body) > 7000:
            anchor = min((int(m["offset"])-begin for m in patient_mentions+phenotype_mentions), default=0)
            body = body[max(0, anchor-1200):max(0, anchor-1200)+7000]
        selected.append((priority, index, f"Passage {index} ({section}): {body}"))
    chosen, used = [], 0
    for _, index, text in sorted(selected, key=lambda x: (-x[0], x[1])):
        remaining = char_budget-used
        if remaining < 200:
            break
        if len(text) > remaining:
            text = text[:remaining]
        chosen.append((index, text));used += len(text)
    return "\n".join(text for _, text in sorted(chosen))


def query_document(doc, entities, char_budget):
    concepts = concepts_of(entities)
    code_map = {f"C{i:03d}": identifier for i, identifier in enumerate(sorted(concepts), 1)}
    lines = []
    for code, identifier in code_map.items():
        surfaces = sorted(set(e["text"] for e in concepts[identifier]))
        passages = []
        for index, passage in enumerate(doc["full_text"]):
            if any(int(passage["offset"]) <= int(e["offset"]) < int(passage["offset"])+len(passage["text"])
                   for e in concepts[identifier]):
                passages.append(index)
        lines.append(f"{code} ({identifier}): " + " / ".join(surfaces[:4]) + "; passages " + ",".join(map(str, passages)))
    content = "Patients:\n"+patient_text(doc)+"\n\nCandidate phenotypes:\n"+"\n".join(lines)+"\n\nClinical evidence:\n"+context_text(doc,entities,char_budget)
    return content, code_map


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train',required=True);ap.add_argument('--input',required=True)
    ap.add_argument('--candidates',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--fewshot',type=int,default=2);ap.add_argument('--context-chars',type=int,default=24000)
    args=ap.parse_args()
    train=read_jsonl(args.train);inputs=read_jsonl(args.input);preds=read_jsonl(args.candidates)
    assert not ({str(d['pmc_id']) for d in train}&{str(d['pmc_id']) for d in inputs})
    assert not any(d.get('entities') or d.get('association') for d in inputs)
    pred_map={str(d['pmc_id']):d for d in preds};assert set(pred_map)=={str(d['pmc_id']) for d in inputs}
    train_texts=[context_text(d,d.get('entities',[]),6000) for d in train]
    vectorizer=TfidfVectorizer(max_features=30000,stop_words='english',ngram_range=(1,2),min_df=1,sublinear_tf=True,dtype=np.float32)
    matrix=vectorizer.fit_transform(train_texts)
    results=[]
    for doc in inputs:
        entities=pred_map[str(doc['pmc_id'])]['entities']
        text,code_map=query_document(doc,entities,args.context_chars)
        q=vectorizer.transform([context_text(doc,entities,6000)])
        scores=(q@matrix.T).toarray()[0]
        nearest=np.argsort(scores)[::-1][:args.fewshot]
        messages=[{'role':'system','content':SYSTEM}]
        for index in nearest:
            ex=train[int(index)]
            example,map_example=query_document(ex,ex.get('entities',[]),2800)
            code_reverse={identifier:code for code,identifier in map_example.items()}
            labels={str(p['patient_id']):[] for p in ex.get('patient',[])}
            for row in ex.get('association',[]):
                labels[str(row['patient_id'])]=sorted(code_reverse[v] for v in row.get('phenotype',[]) if v in code_reverse)
            messages.append({'role':'user','content':example})
            messages.append({'role':'assistant','content':json.dumps(labels,separators=(',',':'))})
        messages.append({'role':'user','content':text})
        results.append({'pmc_id':doc['pmc_id'],'messages':messages,'code_map':code_map,
                        'patient_ids':[str(p['patient_id']) for p in doc.get('patient',[])],
                        'retrieved_training_documents':[str(train[int(i)]['pmc_id']) for i in nearest]})
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in results))
    manifest={'method':'train-only TF-IDF retrieved demonstrations for Qwen3-8B association','configuration':vars(args),
              'documents':len(results),'train_sha256':sha256(args.train),'input_sha256':sha256(args.input),
              'candidates_sha256':sha256(args.candidates),'prompts_sha256':sha256(out),'code_sha256':sha256(__file__)}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'documents':len(results),'prompt_bytes':out.stat().st_size,'sha256':manifest['prompts_sha256']}))


if __name__=='__main__':main()
