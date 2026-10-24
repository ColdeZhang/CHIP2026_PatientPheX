"""Describe existing capped responses without repairing or rescoring any output."""
import argparse
import json
from pathlib import Path
from common import ref,write

def complete_items(body):
    assert body.startswith('[');pos=1;items=[];decoder=json.JSONDecoder()
    while True:
        while pos<len(body) and body[pos].isspace():pos+=1
        if pos==len(body):return items,'awaiting_item_or_closing_bracket',0
        if body[pos]==']':return items,'complete_list_without_eos',len(body)-pos-1
        begin=pos
        try:value,pos=decoder.raw_decode(body,pos)
        except json.JSONDecodeError:return items,'partial_name_json_string',len(body)-begin
        assert isinstance(value,str);items.append(value)
        while pos<len(body) and body[pos].isspace():pos+=1
        if pos==len(body):return items,'awaiting_delimiter',0
        if body[pos]==',':pos+=1;continue
        assert body[pos]==']';return items,'complete_list_without_eos',len(body)-pos-1

def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.collection).parent/'actual';gen=root/'run/infer/generation'
    canonical=[json.loads(v) for v in (gen/'canonical.jsonl').read_text().splitlines()];assert len(canonical)==39
    names_path=Path('/workspace/patientphex-runs/round2-20261023/research/canonical-name-generation-readiness-20261024/public-name-to-id.json')
    names=json.loads(names_path.read_text());rows=[]
    for row in canonical:
        raw=json.loads((gen/f"{row['index']:03d}-raw.json").read_text());ids=json.loads((gen/f"{row['index']:03d}-ids.json").read_text())
        assert len(ids['generated_ids'])==raw['output_tokens']==row['output_tokens']
        if not row['hit_token_limit']:continue
        items,end,tail=complete_items(raw['body']);assert all(v in names for v in items)
        rows.append(dict(row_id=row['row_id'],output_tokens=row['output_tokens'],complete_names=len(items),
            distinct_complete_names=len(set(items)),duplicate_complete_names=len(items)-len(set(items)),
            ending=end,remaining_prefix_characters=tail,
            classification='distinct_legal_name_list_at_cap' if len(items)>1 and len(items)==len(set(items)) else 'other'))
    schema=root/'source/schema_contract.py';text=schema.read_text()
    assert 'used = set()' in text and 'name not in used' in text and "raise ValueError('Duplicate name')" in text
    value=dict(status='EXISTING39_RAW_DIAGNOSTIC_ONLY_NO_REPAIR',collection=ref(a.collection),canonical=ref(gen/'canonical.jsonl'),
        source=ref(__file__),actual_constraint_source=ref(schema),public_names=ref(names_path),
        rows=39,capped=len(rows),cap_rows=rows,natural_eos=sum(x['natural_eos'] for x in canonical),
        natural_eos_strict_rejected=sum(x['natural_eos'] and not x['strict_complete'] for x in canonical),
        generation_already_masks_previously_completed_names=True,training_max_supervised_tokens=269,
        classification_counts={key:sum(x['classification']==key for x in rows) for key in sorted({x['classification'] for x in rows})},
        decoder_prefix_loop_observed=False,decoded_answers_changed=0,new_model_calls=0,new_evaluator_calls=0,
        limitation='Legal distinct long lists do not establish clinical correctness or why EOS probability stayed low.')
    print(json.dumps(write(a.output,value)))

if __name__=='__main__':main()
