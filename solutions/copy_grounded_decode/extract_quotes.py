"""Complete raw39 -> exact copied anchors. No normalization, gold, fit or score."""
import argparse,hashlib,json,os,sys
from collections import Counter
from pathlib import Path
from solutions.grounded_generation.inference.decode import interpret
from solutions.grounded_generation.schema import units


def ref(p):p=Path(p).resolve();raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def checked(item):assert ref(item['path'])==item;return Path(item['path'])
def publish(p,data):
 p=Path(p)
 with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 parent=p.resolve().parent
 while True:
  fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if parent.parent==parent:break
  parent=parent.parent

def save(p,v):publish(p,(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text())
 assert p['schema']=='grounded-extract-complete39-v1'
 # Explicit deny-open applies before checking any supplied artifact.
 forbidden=os.path.realpath(p['gold_metadata_only']['path'])
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0]))==forbidden:raise PermissionError('Gold forbidden during copied-anchor extraction')
 sys.addaudithook(audit)
 for item in p['sources'].values():checked(item)
 assert p['sources']['extract_quotes.py']==ref(__file__)
 for item in p['inputs'].values():checked(item)
 inp=p['inputs'];complete=json.loads(Path(inp['complete']['path']).read_text());wrapper=json.loads(Path(inp['wrapper']['path']).read_text())
 assert wrapper['returncode']==0 and wrapper['timed_out'] is False and 'endpoint_error' not in wrapper
 assert complete['status']=='COMPLETE_FRESH39_CONSTRAINED_RAW_NO_QUALITY' and complete['strict_rows']==39 and complete['EOS_rows']==39 and complete['cap_rows']==0 and complete['rows']==39 and complete['documents']==16
 assert complete['responses']['sha256']==inp['responses']['sha256'] and complete['source']['sha256']==p['sources']['run.py']['sha256']
 docs=[json.loads(x) for x in Path(inp['F1_input']['path']).read_text().splitlines()];by={str(d['pmc_id']):d for d in docs}
 assert len(by)==16 and sum(len(d['patient']) for d in docs)==39 and all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs)
 prompts=[json.loads(x) for x in Path(inp['prompts']['path']).read_text().splitlines()];prompt={r['row_id']:r for r in prompts};records=[json.loads(x) for x in Path(inp['responses']['path']).read_text().splitlines()]
 assert len(prompt)==len(records)==39 and {r['row_id'] for r in records}==set(prompt)
 remote=Path(p['remote_output']);local=Path(p['local_output']).resolve()
 def mapped(item):
  q=local/Path(item['path']).relative_to(remote);assert q.resolve().is_relative_to(local)
  actual=ref(q);assert actual['sha256']==item['sha256'] and actual['bytes']==item['bytes'];return q
 for key in ['generation_refs','raw_refs']:
  assert len(complete[key])==39
  for item in complete[key]:mapped(item)
 generation_by={json.loads(mapped(i).read_text())['row_id']:i for i in complete['generation_refs']};raw_by={json.loads(mapped(i).read_text())['row_id']:i for i in complete['raw_refs']}
 assert set(generation_by)==set(raw_by)==set(prompt)
 out=Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False);quotes=[];decisions=[]
 for record in records:
  row=prompt[record['row_id']];did=str(row['pmc_id']);pid=row['target_patient_id'];doc=by[did]
  assert pid in {x['patient_id'] for x in doc['patient']} and record['pmc_id']==row['pmc_id'] and record['target_patient_id']==pid
  assert row['units']==units(doc) and 'answer' not in row
  generation=json.loads(mapped(generation_by[row['row_id']]).read_text());raw=json.loads(mapped(raw_by[row['row_id']]).read_text())
  assert raw==record and record['raw_decoded'] and not record['parsed'] and record['raw_before_parse'] and record['HF_decode_equal']
  assert record['generation']==generation_by[row['row_id']] and generation['decoded_or_parsed'] is False
  for key,value in generation.items():
   if key!='decoded_or_parsed':assert record[key]==value
  assert len(record['generated_token_ids'])==record['output_tokens']
  decision=interpret(record['response'],doc,record['actual_last_token_eos'],record['hit_token_limit'])
  assert decision['status']=='complete' and not decision['rejected']
  decisions.append(dict(row_id=row['row_id'],pmc_id=did,target_patient_id=pid,raw=raw_by[row['row_id']],decision=decision))
  for index,anchor in enumerate(decision['anchors']):quotes.append(dict(row_id=row['row_id'],pmc_id=did,target_patient_id=pid,anchor_index=index,**anchor))
 publish(out/'quotes.jsonl',b''.join((json.dumps(q,ensure_ascii=False)+'\n').encode() for q in quotes));save(out/'decisions.json',decisions)
 save(out/'manifest.json',dict(plan=ref(a.plan),sources=p['sources'],complete_original39=True,documents=16,patients=39,
  exact_anchors=len(quotes),unique_quotes=len({q['quote'] for q in quotes}),quotes=ref(out/'quotes.jsonl'),decisions=ref(out/'decisions.json'),
  abstaining_rows=sum(d['decision']['status']=='abstain' for d in decisions),rejected_items=sum(len(d['decision']['rejected']) for d in decisions),
  row_reason_counts=dict(Counter(d['decision']['reason'] for d in decisions)),
  rejected_item_reason_counts=dict(Counter(r['reason'] for d in decisions for r in d['decision']['rejected'])),
  valid_empty_arrays=sum(d['decision']['status']=='complete' and d['decision']['generated_items']==0 for d in decisions),
  rows_without_valid_anchor=sum(not d['decision']['anchors'] for d in decisions),
  all_items_rejected_rows=sum(d['decision']['status']=='complete' and d['decision']['generated_items']>0 and not d['decision']['anchors'] for d in decisions),
  EOS_rows=sum(r['actual_last_token_eos'] for r in records),cap_rows=sum(r['hit_token_limit'] for r in records),
  HPO_mapping_or_gold_or_score_performed=False,original_raw_not_repaired=True))
 print(json.dumps(ref(out/'manifest.json')))

if __name__=='__main__':main()
