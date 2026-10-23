#!/usr/bin/env python3
"""Input-selected F1 prompt subset; no task labels, weights, GPU or forward pass."""
from pathlib import Path
import argparse,hashlib,json,datetime

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),bytes=Path(p).stat().st_size,sha256=sha(p))
def save(p,x):
 with p.open('x') as f:f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();b=a.directory.resolve();out=b/'conditional-subset-likelihood-cpu';out.mkdir();source=b/'meta.prompts.jsonl';assert sha(source)=='ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe'
 data=[json.loads(x) for x in source.read_text().splitlines()];seed='sft-json-subset-score-v1';key=lambda text:hashlib.sha256((seed+':'+text).encode()).hexdigest();docs=sorted({str(x['pmc_id']) for x in data},key=key)[:12];chosen=[]
 for did in docs:
  r=min([x for x in data if str(x['pmc_id'])==did],key=lambda x:key(x['row_id']));assert [m['role'] for m in r['messages']]==['system','user'] and 'answer' not in r
  cid=min(r['code_map'],key=lambda cid:key(r['row_id']+':'+cid));chosen.append(dict(prompt=r,selected_cid=cid))
 save(out/'selected-inputs.json',chosen);save(out/'selection-freeze.json',dict(status='12documents_each1prompt_and1CID_frozen_using_only_input',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(__file__),prompts_source=ref(source),seed=seed,selection_rule='pmc hash then row_id hash then CID hash, ascending, first12docs; no outcome/length stratification or exclusions',inputs=ref(out/'selected-inputs.json'),documents=12,prompts=12,CIDs=12,patient_pairs=sum(len(x['prompt']['patient_ids']) for x in chosen),branches_oneCID=sum(2**len(x['prompt']['patient_ids']) for x in chosen),branches_allCIDs=sum(len(x['prompt']['code_map'])*2**len(x['prompt']['patient_ids']) for x in chosen),gold_not_read=True))
 # Prior predicted context read only after answer-independent selection freeze.
 rawpath=b/'meta.responses.jsonl';assert sha(rawpath)=='78e28c724f6cb127383c08a14951ff4cf8381d01e38a9081d91672f8d1488b21';raw={r['row_id']:r for r in map(json.loads,rawpath.read_text().splitlines())};selected=[]
 for c in chosen:
  p=c['prompt'];r=raw[p['row_id']];answer=json.loads(r['response']);assert set(answer)==set(p['code_map']) and not r['hit_token_limit']
  assert all(isinstance(v,list) and len(v)==len(set(v)) and set(v)<=set(p['patient_ids']) for v in answer.values());selected.append(dict(row_id=r['row_id'],predicted_answer=answer))
 save(out/'selected-greedy-context.json',selected);save(out/'preparation.json',dict(selection=ref(out/'selection-freeze.json'),old_prediction=ref(rawpath),selected_context=ref(out/'selected-greedy-context.json'),source=ref(__file__),gold_not_read=True,model_forward=False));print(json.dumps(json.loads((out/'selection-freeze.json').read_text()),indent=2))
if __name__=='__main__':main()
