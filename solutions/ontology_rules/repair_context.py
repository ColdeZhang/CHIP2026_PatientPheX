#!/usr/bin/env python3
"""One format-only retry pass using pre-generated reversible patient codes.

No annotation input is accepted. Valid first responses are immutable; retries
are selected solely by the same strict parser used by the final assembler.
"""
import argparse
import json
from pathlib import Path
import run as baseline
from merge_context import validate_response


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


def main():
    ap=argparse.ArgumentParser()
    for k in ['prompts','responses','encoded-prompts','output-dir']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--retry-responses')
    a=ap.parse_args()
    old=baseline.read_jsonl(a.prompts)
    responses={r['row_id']:r for r in baseline.read_jsonl(a.responses)}
    encoded={r['row_id']:r for r in baseline.read_jsonl(a.encoded_prompts)}
    assert set(responses)==set(encoded)=={r['row_id'] for r in old},'Incomplete original window set'
    bad={};combined=[];retry=[]
    for row in old:
        rid=row['row_id'];new=encoded[rid]
        assert row['code_map']==new['code_map'] and row['evidence_ranges']==new['evidence_ranges']
        assert set(new['patient_id_map'].values())==set(row['patient_ids'])
        try:validate_response(row,responses[rid])
        except (KeyError,ValueError,AssertionError,TypeError) as error:
            bad[rid]={'error':type(error).__name__,'reason':str(error)}
            combined.append(new);retry.append(new)
        else:combined.append(row)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    write_rows(out/'retry-prompts.jsonl',retry)
    write_rows(out/'combined-prompts.jsonl',combined)
    meta={'config':vars(a),'source_sha256':baseline.sha256(__file__),
          'parser_sha256':baseline.sha256(Path(__file__).with_name('merge_context.py')),
          'inputs':{k:baseline.sha256(getattr(a,k)) for k in ['prompts','responses','encoded_prompts']},
          'retry_selection':'strict format/patient/code coverage only, no gold, maximum one retry',
          'failed_rows':bad,'input_windows':len(old),'retry_windows':len(retry),
          'output_hashes':{n:baseline.sha256(out/n) for n in ['retry-prompts.jsonl','combined-prompts.jsonl']}}
    if a.retry_responses:
        repaired={r['row_id']:r for r in baseline.read_jsonl(a.retry_responses)}
        assert set(repaired)==set(bad),'Retry windows differ from format failures'
        final=[];origins={}
        for row in combined:
            rid=row['row_id'];response=repaired[rid] if rid in bad else responses[rid]
            validate_response(row,response)
            final.append(response);origins[rid]='retry' if rid in bad else 'original'
        write_rows(out/'combined-responses.jsonl',final)
        meta['inputs']['retry_responses']=baseline.sha256(a.retry_responses)
        meta['response_origins']=origins
        meta['output_hashes']['combined-responses.jsonl']=baseline.sha256(out/'combined-responses.jsonl')
    (out/'repair.manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta))


if __name__=='__main__':main()
