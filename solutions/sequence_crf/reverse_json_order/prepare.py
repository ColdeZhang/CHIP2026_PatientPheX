#!/usr/bin/env python3
"""CPU-only fixed reverse ordering, exact ID inverse and native token inventory."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,re,signal,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.path.append('/usr/lib/python3/dist-packages')
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

DIVIDER='\n\nCandidate phenotypes:\n'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_bytes().splitlines()]
def save(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def digest(s):return hashlib.sha256(s.encode()).hexdigest()

def parts(raw):
 """Retain exact objects and separators; only top-level code values may change."""
 d=json.JSONDecoder();start=len(raw)-len(raw.lstrip());assert raw[start]=='['
 pos=start+1;lead=re.match(r'\s*',raw[pos:]).group();pos+=len(lead);objects=[];seps=[]
 if raw[pos:pos+1]==']':return raw[:pos],[],[],raw[pos:]
 while True:
  obj,end=d.raw_decode(raw,pos);assert isinstance(obj,dict);objects.append((raw[pos:end],obj))
  m=re.match(r'\s*',raw[end:]);nextpos=end+len(m.group())
  if raw[nextpos:nextpos+1]==']':return raw[:start+1]+lead,objects,seps,raw[end:]
  assert raw[nextpos:nextpos+1]==','
  nxt=nextpos+1;nxt+=len(re.match(r'\s*',raw[nxt:]).group());seps.append(raw[end:nxt]);pos=nxt

def replace_top_code(raw,new):
 """Use JSON token boundaries, not substitution in quote/context or substrings."""
 decoder=json.JSONDecoder();i=len(raw)-len(raw.lstrip());assert raw[i]=='{';i+=1
 while True:
  i+=len(re.match(r'\s*',raw[i:]).group());key,end=decoder.raw_decode(raw,i);assert isinstance(key,str)
  i=end;i+=len(re.match(r'\s*',raw[i:]).group());assert raw[i]==':';i+=1;i+=len(re.match(r'\s*',raw[i:]).group())
  value,end=decoder.raw_decode(raw,i)
  if key=='code':assert isinstance(value,str);return raw[:i]+json.dumps(new,ensure_ascii=False)+raw[end:]
  i=end;i+=len(re.match(r'\s*',raw[i:]).group());assert raw[i]==',';i+=1

def reverse(row):
 assert [m['role'] for m in row['messages']]==['system','user']
 prefix,raw=row['messages'][1]['content'].split(DIVIDER,1)
 lead,objs,seps,tail=parts(raw);codes=list(row['code_map']);assert [v['code'] for _,v in objs]==codes and len(set(codes))==len(codes)
 assert not seps or len(set(seps))==1,'No format normalization permitted'
 mapping={old:new for (r,v),new in zip(reversed(objs),codes) for old in [v['code']]}
 rev=[replace_top_code(r,mapping[v['code']]) for r,v in reversed(objs)]
 newraw=lead+(seps[0] if seps else '').join(rev)+tail
 result=copy.deepcopy(row);result['messages'][1]['content']=prefix+DIVIDER+newraw
 result['code_map']={mapping[v['code']]:row['code_map'][v['code']] for _,v in reversed(objs)}
 assert result['messages'][0]==row['messages'][0] and result['patient_ids']==row['patient_ids']
 assert {k:v for k,v in result.items() if k not in ('code_map','messages')}=={k:v for k,v in row.items() if k not in ('code_map','messages')}
 old_by_code={v['code']:{k:x for k,x in v.items() if k!='code'} for _,v in objs}
 for v in json.loads(newraw):assert {k:x for k,x in v.items() if k!='code'}==old_by_code[{v:k for k,v in mapping.items()}[v['code']]]
 return result,mapping

def contracts():
 checked=0
 for n in [0,1,2,3,8,11]:
  pool=[dict(code='C%03d'%i,meaning='M',evidence=[{'context':'literal C000 / C010 and "code": "C001" / [PATIENT P1: x]', 'quote':'P1 C000'}]) for i in range(n)]
  r=dict(row_id='synthetic',pmc_id='input',code_map={p['code']:'HP:%07d'%(i+1) for i,p in enumerate(pool)},patient_ids=['P1','P10'],messages=[{'role':'system','content':'fixed'},{'role':'user','content':'P1: literal C000 P10'+DIVIDER+json.dumps(pool,ensure_ascii=False)+'\n'}])
  a,m=reverse(r);z,mi=reverse(a);assert z==r and all(mi[v]==k for k,v in m.items());checked+=1
 return checked

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU>30s')));signal.alarm(30);tick=time.monotonic()
 p=load(b/'plan.json');assert sha(__file__)==p['source_sha256']
 for r in p['references'].values():assert sha(r['path'])==r['sha256']
 r=p['references'];original=rows(r['prompts']['path']);assert len(original)==80
 oldmeta={x['row_id']:{k:x[k] for k in ['input_tokens','output_tokens','batch_seconds']} for x in rows(r['old_responses_input_cost_only']['path'])}
 env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True);env.globals['raise_exception']=lambda x:(_ for _ in ()).throw(ValueError(x));template=env.from_string(Path(r['template']['path']).read_text());tok=Tokenizer.from_file(r['tokenizer']['path'])
 def encode(ms):
  s=template.render(messages=ms,tools=None,add_generation_prompt=True,enable_thinking=False);return s,len(tok.encode(s,add_special_tokens=False).ids)
 new=[];audit=[];total=0;changed=0
 for row in original:
  z,m=reverse(row);back,inv=reverse(z);assert back==row and back['messages'][1]['content'].encode()==row['messages'][1]['content'].encode()
  # Exact original patient/atomic-HPO meanings round-trip for synthetic responses.
  fake={c:list(row['patient_ids']) for c in row['code_map']};mapped={m[k]:v for k,v in fake.items()};assert {inv[k]:v for k,v in mapped.items()}==fake
  oldtext,oldn=encode(row['messages']);newtext,n=encode(z['messages']);assert oldn==oldmeta[row['row_id']]['input_tokens'];assert n<=14000
  new.append(z);total+=n;changed+=row!=z
  audit.append(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],old_to_new=m,new_to_old=inv,candidates=len(m),original_input_tokens=oldn,new_input_tokens=n,delta_tokens=n-oldn,full_render_sha256=digest(newtext),original_render_sha256=digest(oldtext),inverse_all_fields_and_user_byteexact=True,semantic_candidate_fields_exact=True))
 with (b/'prompts.jsonl').open('x') as f:
  for row in new:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 save(b/'inverse-and-token-audit.json',audit)
 oldwall=load(r['old_manifest']['path'])['elapsed_seconds'];oldtotal=sum(v['input_tokens'] for v in oldmeta.values());outtokens=sum(v['output_tokens'] for v in oldmeta.values())
 report=dict(scope='Input-only readiness; no forward, new score, gold or training',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(__file__),plan=ref(b/'plan.json'),rows=80,documents=len({x['pmc_id'] for x in original}),concepts=sum(len(x['code_map']) for x in original),changed_rows=changed,unchanged_rows=80-changed,all_roundtrip_byteexact=True,synthetic_contracts=contracts(),original_total_tokens=oldtotal,new_total_tokens=total,maximum_new_tokens=max(x['new_input_tokens'] for x in audit),token_changed_rows=sum(x['delta_tokens']!=0 for x in audit),maximum_with_1600_output=max(x['new_input_tokens'] for x in audit)+1600,above14000=0,truncated=0,cost=dict(old80_actual_seconds=oldwall,ratio_with_old_output=(total+outtokens)/(oldtotal+outtokens),proxy_seconds=oldwall*(total+outtokens)/(oldtotal+outtokens),caveat='Estimate only; reversed generation may change length. Same original SFT32 model, optional future budget600+30.'),artifacts={n:ref(b/n) for n in ['prompts.jsonl','inverse-and-token-audit.json']},cpu_seconds=time.monotonic()-tick)
 save(b/'report.json',report);print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
