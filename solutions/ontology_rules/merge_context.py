#!/usr/bin/env python3
"""Validate LLM patient/code choices and preserve the frozen entity predictions."""
import argparse,json,re
from pathlib import Path
import run as baseline


def parse(text):
    text=text.strip();text=re.sub(r'^```(?:json)?\s*|\s*```$','',text)
    return json.loads(text)


def main():
    ap=argparse.ArgumentParser()
    for k in ['input','base','prompts','responses','output']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--fallback-rules',action='store_true')
    a=ap.parse_args();inputs=baseline.read_jsonl(a.input);base={d['pmc_id']:d for d in baseline.read_jsonl(a.base)};prompts=baseline.read_jsonl(a.prompts);responses={r['row_id']:r for r in baseline.read_jsonl(a.responses)}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    assert {d['pmc_id'] for d in inputs}==set(base)
    result={d['pmc_id']:{str(p['patient_id']):set() for p in d['patient']} for d in inputs};invalid=[];accepted=[];covered=set()
    for row in prompts:
        pmc=row['pmc_id'];cmap=row['code_map'];rid=row['row_id'];covered.add(pmc)
        try:
            response=responses[rid];assert not response.get('hit_token_limit',False),'truncated'
            obj=parse(response['response']);assert isinstance(obj,dict) and set(obj)==set(row['patient_ids']),'patient coverage'
            for pid,codes in obj.items():
                assert isinstance(codes,list) and all(isinstance(c,str) and c in cmap for c in codes),'candidate validity'
            for pid,codes in obj.items():result[pmc][pid].update(cmap[c] for c in codes)
            accepted.append(rid)
        except (KeyError,ValueError,AssertionError,TypeError) as error:
            invalid.append({'row_id':rid,'reason':type(error).__name__})
            if not a.fallback_rules:continue
            allowed=set(cmap.values())
            for assoc in base[pmc]['association']:result[pmc][str(assoc['patient_id'])].update(set(assoc['phenotype'])&allowed)
    if invalid and not a.fallback_rules:raise ValueError(f'{len(invalid)} invalid or incomplete LLM windows; refusing silent partial output')
    preds=[dict(base[d['pmc_id']],association=[{'patient_id':p['patient_id'],'phenotype':sorted(result[d['pmc_id']][str(p['patient_id'])])} for p in d['patient']]) for d in inputs]
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in preds))
    manifest={'config':vars(a),'source_sha256':baseline.sha256(__file__),'inputs':{k:baseline.sha256(getattr(a,k)) for k in ['input','base','prompts','responses']},'accepted_windows':len(accepted),'invalid_windows':invalid,'fallback_used':bool(invalid and a.fallback_rules),'documents_without_candidate_windows':len(inputs)-len(covered),'output_sha256':baseline.sha256(output)}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
