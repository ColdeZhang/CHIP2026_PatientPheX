#!/usr/bin/env python3
"""Input-only exact-coordinate/window census; no model, labels, fit or quality."""
from pathlib import Path
import argparse,collections,hashlib,json,math,re,signal,time,datetime,os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
from tokenizers import Tokenizer

def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_bytes().splitlines()]
def ids_only(p):return set(re.findall(r'"pmc_id"\s*:\s*"([^"\n]+)"',Path(p).read_text()))
def save(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def chunks(n):
 out=[];start=0
 while start<n:
  end=min(n,start+510);out.append((start,end))
  if end==n:break
  start=end-128
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory;p=load(b/'plan.json');tick=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU120')));signal.alarm(120)
 assert ref(__file__)['sha256']==p['source_sha256']
 for r in p['references'].values():assert ref(r['path'])==r
 meta=load(p['references']['sft_preparation']['path']);trainids=set(meta['training_document_ids']);heldids=ids_only(p['references']['F1_input']['path']);assert len(trainids)==32 and len(heldids)==16 and not trainids&heldids
 tok=Tokenizer.from_file(p['references']['tokenizer']['path']);tok.no_truncation();tok.no_padding()
 # Tokenizer saved by old CE, but vocabulary/normalizer must match the PUBLIC base.
 td=json.loads(tok.to_str());vocab=Path(p['references']['vocab']['path']).read_text().splitlines();assert td['model']['vocab']=={x:i for i,x in enumerate(vocab)} and td['normalizer']['lowercase']
 specs=[]
 for x in meta['provenance']:
  for k in ['target_input','candidates','assembly_manifest','semantic_manifest','CRF_manifest','generator_train','prompts']:assert ref(x[k]['path'])==x[k]
  gi=ids_only(x['generator_train']['path']);ti=ids_only(x['target_input']['path']);assert gi|ti==trainids and not gi&ti and not (gi|ti)&heldids
  am=load(x['assembly_manifest']['path']);assert am['input_annotations_used'] is False
  specs.append(dict(split='train',input=x['target_input'],candidates=x['candidates'],prompts=x['prompts'],generator_ids=sorted(gi),target_ids=sorted(ti)))
 am=load(p['references']['F1_candidates_manifest']['path']);assert am['inputs']['train']==ref(am['config']['train'])['sha256'] and ids_only(am['config']['train'])==trainids
 assert am['inputs']['input']==p['references']['F1_input']['sha256'] and am['output_sha256']==p['references']['F1_candidates']['sha256']
 specs.append(dict(split='F1_input_only',input=p['references']['F1_input'],candidates=p['references']['F1_candidates'],prompts=p['references']['F1_prompts'],generator_ids=sorted(trainids),target_ids=sorted(heldids)))
 all_docs=[];occurrence_rows=[];pair_rows=[];seen=set();totals=collections.defaultdict(collections.Counter)
 for spec in specs:
  split=spec['split'];ds=rows(spec['input']['path']);pred={str(x['pmc_id']):x for x in rows(spec['candidates']['path'])};prompts=rows(spec['prompts']['path']);declared=collections.defaultdict(set)
  for row in prompts:
   for concept in row['code_map'].values():
    for patient in row['patient_ids']:declared[str(row['pmc_id'])].add((str(patient),concept))
  for doc in ds:
   assert not {'entities','association'}&doc.keys();pmc=str(doc['pmc_id']);assert pmc not in seen;seen.add(pmc)
   encoded=[tok.encode(x['text'],add_special_tokens=False) for x in doc['full_text']];off=[e.offsets for e in encoded];lens=[len(e.ids) for e in encoded];starts=[];n=0
   for z in lens:starts.append(n);n+=z
   windows=[]
   for pi,z in enumerate(lens):
    for left,right in chunks(z):windows.append(dict(passage_index=pi,left=left,right=right,tokens=right-left+2))
   patient=collections.defaultdict(list);concepts=collections.defaultdict(list);bad=[]
   def add(kind,key,x):
    begin=int(x['offset']);end=begin+int(x['length']);match=[i for i,z in enumerate(doc['full_text']) if z['offset']<=begin<end<=z['offset']+len(z['text'])]
    assert len(match)==1,(pmc,kind,begin,end)
    pi=match[0];passage=doc['full_text'][pi];left=begin-passage['offset'];right=end-passage['offset'];assert passage['text'][left:right]==x['text'],(pmc,kind,x)
    ts=[i for i,(l,r) in enumerate(off[pi]) if l<right and left<r];assert ts
    a0,z0=ts[0],ts[-1]+1;ws={i for i,w in enumerate(windows) if w['passage_index']==pi and w['left']<=a0<z0<=w['right']}
    assert ws,(pmc,'no full span window',x)
    support=set()
    for i in ws:
     w=windows[i];support.update(range(starts[pi]+w['left'],starts[pi]+w['right']))
    item=dict(passage=pi,token_start=a0,token_end=z0,windows=ws,support=support)
    aligned=off[pi][a0][0]==left and off[pi][z0-1][1]==right
    occurrence_rows.append(dict(split=split,pmc_id=pmc,kind=kind,key=key,offset=begin,length=end-begin,text=x['text'],passage_index=pi,token_start=a0,token_end=z0,token_boundary_exact=aligned,window_ids=sorted(ws)))
    totals[split][kind+'_occurrences']+=1;totals[split][kind+'_not_exact_token_boundary']+=not aligned;totals[split][kind+'_outside_first1022_rawtokens']+=starts[pi]+z0>1022
    return item
   for pat in doc['patient']:
    assert str(pat['patient_id']) not in patient
    patient[str(pat['patient_id'])]=[add('patient',str(pat['patient_id']),x) for x in pat.get('mention',[])]
   for x in pred[pmc]['entities']:
    keys=[x['text']] if str(x['identifier'])=='-1' else str(x['identifier']).split(';')
    for k in keys:concepts[k].append(add('phenotype',k,x))
   actual={(pid,k) for pid in patient for k in concepts};assert actual==declared[pmc]
   counts=collections.Counter()
   for pid,k in sorted(actual):
    pp,cc=patient[pid],concepts[k];pw=set().union(*(x['windows'] for x in pp)) if pp else set();cw=set().union(*(x['windows'] for x in cc));ps=set().union(*(x['support'] for x in pp)) if pp else set();cs=set().union(*(x['support'] for x in cc))
    common=pw&cw;support=ps&cs;samepass=bool({x['passage'] for x in pp}&{x['passage'] for x in cc});counts['pairs']+=1;counts['pairs_without_patient_mention']+=not pp;counts['pairs_no_same_window']+=not common;counts['pairs_no_shared_attention_support']+=not support;counts['pairs_only_cross_passage']+=not samepass;counts['pair_window_incidence']+=len(common)
    pair_rows.append(dict(split=split,pmc_id=pmc,patient_id=pid,concept=k,patient_occurrences=len(pp),phenotype_occurrences=len(cc),co_windows=len(common),shared_token_support=len(support),same_passage=samepass))
   d=dict(split=split,pmc_id=pmc,passages=len(lens),raw_tokens=n,documents_over1022=int(n>1022),windows=len(windows),window_tokens=sum(x['tokens'] for x in windows),pad512_tokens=512*len(windows),max_window_tokens=max(x['tokens'] for x in windows),patients=len(patient),concepts=len(concepts),**counts)
   # Global dense attention is unnecessary; these are memory scale diagnostics only.
   d['dense_document_attention_fp32_bytes_12heads']=12*n*n*4;d['last_layer_one512_attention_fp32_bytes_12heads']=12*512*512*4;all_docs.append(d)
   for key,value in d.items():
    if isinstance(value,int) and key not in ['max_window_tokens','dense_document_attention_fp32_bytes_12heads','last_layer_one512_attention_fp32_bytes_12heads']:totals[split][key]+=value
   totals[split]['documents']+=1
 assert len(seen)==48
 for name,data in [('documents',all_docs),('occurrences',occurrence_rows),('pairs',pair_rows)]:save(b/(name+'.json'),data)
 report=dict(status='CPU_INPUT_ONLY_NO_TRAIN_OR_QUALITY',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(__file__),plan=ref(b/'plan.json'),seconds=time.monotonic()-tick,totals=totals,max_document_tokens=max(x['raw_tokens'] for x in all_docs),max_windows_per_document=max(x['windows'] for x in all_docs),max_dense_attention_bytes=max(x['dense_document_attention_fp32_bytes_12heads'] for x in all_docs),references=p['references'],strict_sources=specs,tokenizer=dict(no_truncation=True,no_padding=True,public_vocab30522_exact=True,wordpiece_boundary_mismatch_policy='Keep original character coordinates, pool overlapping token range; report mismatch, do not extend entity output.'),artifacts={n:ref(b/(n+'.json')) for n in ['documents','occurrences','pairs']},label_reads=False,encoder_forward=False)
 save(b/'report.json',report);print(json.dumps({k:report[k] for k in ['seconds','totals','max_document_tokens','max_windows_per_document','max_dense_attention_bytes']}))
if __name__=='__main__':main()
