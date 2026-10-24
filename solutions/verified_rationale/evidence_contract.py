"""Mechanical validation only. Passing this does not establish patient attribution."""
import json

KEYS={'supported','patient_anchor','phenotype_anchor','bridge','rationale'}
def unique(items):
 d={}
 for k,v in items:
  if k in d:raise ValueError('duplicate_JSON_key')
  d[k]=v
 return d

def final_json(response_without_terminal_eos, *, eos_observed, stopped_at_cap):
 if not eos_observed or stopped_at_cap:raise ValueError('incomplete_generation')
 text=response_without_terminal_eos.strip()
 if '<think>' in text or '</think>' in text:
  if not text.startswith('<think>') or text.count('<think>')!=1 or text.count('</think>')!=1:raise ValueError('invalid_think_boundary')
  text=text.split('</think>',1)[1].strip()
 obj=json.loads(text,object_pairs_hook=unique)
 if not isinstance(obj,dict) or set(obj)!=KEYS:raise ValueError('final_schema')
 if type(obj['supported']) is not bool or not isinstance(obj['rationale'],str) or len(obj['rationale'].split())>80:raise ValueError('boolean_or_rationale')
 if not isinstance(obj['bridge'],list) or len(obj['bridge'])>3:raise ValueError('bridge_shape')
 return obj

def locate(triple,units):
 if not isinstance(triple,list) or len(triple)!=3:raise ValueError('quote_shape')
 uid,quote,occ=triple
 if not isinstance(uid,str) or uid not in units or not isinstance(quote,str) or not quote or type(occ) is not int or occ<0:raise ValueError('quote_type')
 u=units[uid];hits=[];start=0
 while True:
  k=u['text'].find(quote,start)
  if k<0:break
  hits.append(k);start=k+1
 if occ>=len(hits):raise ValueError('quote_absent')
 return dict(unit_id=uid,offset=u['offset']+hits[occ],length=len(quote),quote=quote,occurrence=occ)

def contains(span,mention):return span['offset']<=mention['offset'] and mention['offset']+mention['length']<=span['offset']+span['length']
def validate(obj, *, units, target_mentions, concept_mentions):
 if not obj['supported']:
  if obj['patient_anchor']!=[] or obj['phenotype_anchor']!=[] or obj['bridge']!=[]:raise ValueError('abstain_has_evidence')
  return dict(status='TEACHER_ABSTAIN',semantic_support='not_assessed')
 patient=locate(obj['patient_anchor'],units);phenotype=locate(obj['phenotype_anchor'],units);bridge=[locate(t,units) for t in obj['bridge']]
 if not any(contains(patient,m) for m in target_mentions):raise ValueError('anchor_not_given_target')
 if not any(contains(phenotype,m) for m in concept_mentions):raise ValueError('phenotype_not_selected_training_concept')
 if len({(b['unit_id'],b['offset'],b['length']) for b in bridge})!=len(bridge):raise ValueError('duplicate_bridge')
 return dict(status='MECHANICAL_PASS_SEMANTIC_REVIEW_REQUIRED',patient_anchor=patient,phenotype_anchor=phenotype,bridge=bridge,rationale=obj['rationale'],semantic_support='not_assessed',gold_hint_is_not_prediction_accuracy=True)

def synthetic_contract():
 units={'U0000':dict(offset=10,text='P1 and P2 both had ataxia. P1 alone had seizures. café café.')}
 good=dict(supported=True,patient_anchor=['U0000','P1',0],phenotype_anchor=['U0000','ataxia',0],bridge=[['U0000','P1 and P2 both had ataxia.',0]],rationale='The statement attributes ataxia to both listed patients, including P1.')
 def run(x,**kw):return validate(final_json(x,**kw),units=units,target_mentions=[dict(offset=10,length=2)],concept_mentions=[dict(offset=29,length=6)])
 assert run('<think>analysis</think>'+json.dumps(good),eos_observed=True,stopped_at_cap=False)['status']=='MECHANICAL_PASS_SEMANTIC_REVIEW_REQUIRED'
 # The arithmetic positions are checked against literal slices, not presumed by labels.
 assert units['U0000']['text'][19:25]=='ataxia'
 assert locate(['U0000','café',1],units)['offset']==65
 errors=[]
 variants=[('cap',json.dumps(good),dict(eos_observed=True,stopped_at_cap=True)),('noEOS',json.dumps(good),dict(eos_observed=False,stopped_at_cap=False)),('unclosed','<think>'+json.dumps(good),dict(eos_observed=True,stopped_at_cap=False)),('fence','```json\n'+json.dumps(good)+'\n```',dict(eos_observed=True,stopped_at_cap=False)),('trailing',json.dumps(good)+' explanation',dict(eos_observed=True,stopped_at_cap=False)),('duplicate','{"supported":true,'+json.dumps(good)[1:],dict(eos_observed=True,stopped_at_cap=False))]
 for name,field,value in [('foreign','patient_anchor',['U0000','P2',0]),('wrongconcept','phenotype_anchor',['U0000','seizures',0]),('unknownunit','bridge',[['U0001','ataxia',0]]),('boolocc','bridge',[['U0000','ataxia',True]]),('badocc','bridge',[['U0000','ataxia',1]]),('hallucinated','bridge',[['U0000','not in source',0]]),('duplicatebridge','bridge',[['U0000','ataxia',0],['U0000','ataxia',0]]),('longreason','rationale',' '.join(['word']*81))]:
  obj=dict(good);obj[field]=value;variants.append((name,json.dumps(obj),dict(eos_observed=True,stopped_at_cap=False)))
 for name,text,kw in variants:
  try:run(text,**kw)
  except (ValueError,json.JSONDecodeError):errors.append(name)
  else:raise AssertionError('accepted '+name)
 assert len(errors)==14
 abstain=dict(supported=False,patient_anchor=[],phenotype_anchor=[],bridge=[],rationale='Insufficient support.')
 assert run(json.dumps(abstain),eos_observed=True,stopped_at_cap=False)['status']=='TEACHER_ABSTAIN'
 return dict(supported_case_passed=True,abstention_passed=True,distinct_rejections=errors,semantic_truth_automatically_claimed=False)
if __name__=='__main__':print(json.dumps(synthetic_contract(),indent=2))
