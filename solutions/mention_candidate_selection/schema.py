"""Closed candidate subset + missing / NIL / NO; finite native-token grammar."""
import itertools,json
SYSTEM='''Normalize only the marked clinical mention in its supplied context.
Choose from the supplied C01..C08 candidates; their order is retrieval rank, not an answer.
Use one of exactly three JSON modes:
{"selected":["C01",...],"missing":false} for a positive HPO-representable phenotype: select every equivalent supplied concept in candidate order. Set missing true if any explicitly expressed correct HPO concept is absent from the options; [] with missing true is allowed. Do not guess another option to replace a missing concept.
{"NIL":true} for a positive phenotype without an equivalent HPO concept.
{"NO":true} for a negated/normal finding or non-phenotype.
Keep specificity, direction, anatomy, and negation. Do not infer findings from a disease. Return no extra keys, explanation, or identifiers. Never select duplicate options. An empty selected list with missing false is invalid.'''

def serialize(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'))

def strict_parse(raw,candidates,*,eos,capped):
 if not eos or capped:raise ValueError('Incomplete output')
 def obj(pairs):
  result={}
  for k,v in pairs:
   if k in result:raise ValueError('Duplicate object key')
   result[k]=v
  return result
 value=json.loads(raw,object_pairs_hook=obj)
 if type(value)is not dict:raise ValueError('Expected object')
 if set(value)=={'NIL'} and value['NIL'] is True:return {'status':'NIL','selected':[],'ids':['-1']}
 if set(value)=={'NO'} and value['NO'] is True:return {'status':'NO','selected':[],'ids':[]}
 if set(value)!={'selected','missing'}:raise ValueError('Unknown or missing mode keys')
 selected=value['selected'];missing=value['missing'];codes=[x['code'] for x in candidates]
 if type(selected)is not list or type(missing)is not bool or any(type(x)is not str for x in selected):raise ValueError('Invalid selected/missing types')
 if len(selected)!=len(set(selected)) or any(x not in codes for x in selected):raise ValueError('Unknown or duplicate candidate code')
 if selected!=[x for x in codes if x in selected]:raise ValueError('Candidate selection order differs')
 if not selected and not missing:raise ValueError('Empty HPO without missing is not NO')
 return {'status':'HPO','selected':selected,'ids':[x['identifier'] for x in candidates if x['code'] in selected],'missing':missing}

class CandidateCodec:
 def __init__(self,tokenizer,eos_id,count=8):
  assert 0<=count<=8;self.eos_id=eos_id;self.root={};self.outputs=[];codes=[f'C{i:02d}' for i in range(1,count+1)]
  values=[{'NIL':True},{'NO':True}]
  for n in range(count+1):
   for selected in itertools.combinations(codes,n):
    for missing in (False,True):
     if selected or missing:values.append({'selected':list(selected),'missing':missing})
  for value in values:
   text=serialize(value);ids=tokenizer.encode(text,add_special_tokens=False).ids
   assert tokenizer.decode(ids,skip_special_tokens=False)==text
   node=self.root
   for token in ids:node=node.setdefault(token,{})
   assert eos_id not in node;node[eos_id]={eos_id:{}};self.outputs.append({'value':value,'text':text,'ids':ids})
  assert len(self.outputs)==2**(count+1)+1
 def allowed(self,prefix):
  node=self.root
  for token in list(prefix):
   if token not in node:raise ValueError('Prefix outside complete finite grammar')
   node=node[token]
  if not node:raise ValueError('Tokens after terminal EOS')
  return sorted(node)
 def callback(self,prompt_width):
  def f(batch_id,input_ids):
   values=input_ids.tolist() if hasattr(input_ids,'tolist') else list(input_ids)
   return self.allowed(values[prompt_width:])
  return f
