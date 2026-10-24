"""Exact old candidate reversal plus untouched full-report suffix; no tokenizer/model."""

import copy

import json

import re

DIVIDER='\n\nCandidate phenotypes:\n'

APPEND_PREFIX='\n\nFull supplied report, original passage order:\n'

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


def fulltext_reverse(full, short):
    """Reverse only the exact old candidate prefix; never parse/rebuild the suffix."""
    assert 'answer' not in full and 'answer' not in short
    assert set(full) == set(short) == {'row_id', 'pmc_id', 'code_map', 'patient_ids', 'messages'}
    assert {k:v for k,v in full.items() if k != 'messages'} == {k:v for k,v in short.items() if k != 'messages'}
    assert full['messages'][0] == short['messages'][0]
    old = short['messages'][1]['content']; expanded = full['messages'][1]['content']
    assert expanded.startswith(old)
    suffix = expanded[len(old):]
    assert suffix.startswith(APPEND_PREFIX) and len(suffix) > len(APPEND_PREFIX)
    reversed_short, mapping = reverse(short)
    result = copy.deepcopy(full)
    result['code_map'] = reversed_short['code_map']
    result['messages'][1]['content'] = reversed_short['messages'][1]['content'] + suffix
    assert result['messages'][1]['content'][len(reversed_short['messages'][1]['content']):].encode() == suffix.encode()
    assert result['messages'][0] == full['messages'][0] and result['patient_ids'] == full['patient_ids']
    assert sorted(result['code_map'].values()) == sorted(full['code_map'].values())
    return result, reversed_short, mapping, suffix
