#!/usr/bin/env python3
"""Rechunk structurally failed quote requests once, without reading evaluation labels."""
import argparse
import json
from pathlib import Path
from prepare_extraction import windows
from normalize_extraction import parse_json
from run import sha256

def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def write_rows(path,rows):
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prompts',required=True);ap.add_argument('--responses',required=True)
    ap.add_argument('--output-dir',required=True);ap.add_argument('--retry-responses')
    ap.add_argument('--chunk-chars',type=int,default=1800)
    args=ap.parse_args();prompts=read_jsonl(args.prompts);responses=read_jsonl(args.responses)
    by_id={x['row_id']:x for x in responses};assert len(by_id)==len(responses)
    assert {x['row_id'] for x in prompts}==set(by_id)
    failed={x['row_id'] for x in responses if x.get('hit_token_limit') or not isinstance(parse_json(x['response']),list)}
    retries=[]
    for prompt in prompts:
        if prompt['row_id'] not in failed:continue
        doc={'pmc_id':prompt['pmc_id'],'full_text':[{'text':s['text'],'offset':s['offset'],'section_type':s.get('section','')} for s in prompt['segments']]}
        for chunk in windows(doc,args.chunk_chars,100):
            retries.append(dict(chunk,row_id=prompt['row_id']+f":retry:{chunk['chunk_index']}",source_row_id=prompt['row_id'],retry_number=1,
                                messages=prompt['messages'][:-1]+[{'role':'user','content':'TEXT:\n'+chunk['text']}]))
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    write_rows(output/'retry-prompts.jsonl',retries)
    manifest={'criterion':'only missing/invalid JSON array or token-limit response; no evaluation labels','configuration':vars(args),
              'failed_original_row_ids':sorted(failed),'retry_requests':len(retries),'code_sha256':sha256(__file__),
              'input_files':{k:sha256(getattr(args,k)) for k in ['prompts','responses']}}
    if args.retry_responses:
        rr=read_jsonl(args.retry_responses);assert {x['row_id'] for x in rr}=={x['row_id'] for x in retries} and len(rr)==len(retries)
        combined_prompts=[p for p in prompts if p['row_id'] not in failed]+retries
        combined_responses=[r for r in responses if r['row_id'] not in failed]+rr
        write_rows(output/'combined-prompts.jsonl',combined_prompts);write_rows(output/'combined-responses.jsonl',combined_responses)
        manifest['retry_responses_sha256']=sha256(args.retry_responses)
        manifest['remaining_invalid_or_truncated']=sum(x.get('hit_token_limit') or not isinstance(parse_json(x['response']),list) for x in rr)
        manifest['output_files']={name:sha256(output/name) for name in ['retry-prompts.jsonl','combined-prompts.jsonl','combined-responses.jsonl']}
    (output/'retry-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
