"""New explicit response-code representation; untouched clinical evidence prefix."""
import copy,json
OLD_OUTPUT='Return one JSON object mapping EVERY candidate code to a list of supplied patient IDs. Return [] when no supplied patient is supported. Use only listed candidate codes and patient IDs.'
NEW_OUTPUT='Return one JSON object mapping EVERY candidate code to a list of allowed response patient codes from the response-code table. Return [] when no supplied patient is supported. Use only listed candidate codes and response patient codes; do not output original patient IDs.'
OLD_RELATIVE="A supplied patient can be a relative; report that relative's own findings for their patient ID."
NEW_RELATIVE="A supplied patient can be a relative; report that relative's own findings using that relative's mapped response patient code."
CODE_INSTRUCTION='Response codes are output-only handles, not clinical case numbers. The original patient_ids metadata, global anchors, [PATIENT original_id: text] markers and clinical identifiers in the report remain unchanged identity attributes. O-prefixed identifiers can encode clinical case numbers; P-prefixed identifiers are appearance-order identities, not clinical numbers. Match the unchanged original patient identity in the evidence, then translate it using the appended response-code table. Never merge different identities because their mention text is identical. This is a new prompt/output representation, not a permutation of clinical identities.'
TABLE_PREFIX='\n\nResponse-code table (output handles only; original evidence above is unchanged):\n'
REPRESENTATION='normal-CID-to-independent-patient-response-code-v1'

def unique_object(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('Duplicate JSON key')
  d[k]=v
 return d

def parse_matrix(text, codes, patients):
 data=json.loads(text,object_pairs_hook=unique_object)
 if not isinstance(data,dict) or set(data)!=set(codes):raise ValueError('Require every and only candidate CID')
 for k,v in data.items():
  if not isinstance(v,list) or any(type(x) is not str for x in v):raise ValueError('Require list of string patient codes')
  if len(set(v))!=len(v):raise ValueError('Duplicate patient response code')
  if set(v)-set(patients):raise ValueError('Unknown patient response code')
 return data

def mapping(row):
 ids=row['patient_ids'];cids=list(row['code_map'])
 if not ids or len(ids)!=len(set(ids)) or any(type(x) is not str for x in ids):raise ValueError('Duplicate/invalid original patient ID')
 codes=[f'PT{i:03d}' for i in range(1,len(ids)+1)]
 if set(codes)&(set(ids)|set(cids)):raise ValueError('Response code collides with original PID or CID')
 # Reject literal alias collisions in the unchanged evidence, rather than silently
 # change the namespace/order for a difficult document.
 if any(code in row['messages'][-1]['content'] for code in codes):raise ValueError('Response code already occurs in original evidence')
 return dict(zip(codes,ids))

def decode_response(text,row):
 m=row['response_code_map']
 if list(m)!=row['response_patient_codes'] or list(m.values())!=row['patient_ids'] or len(set(m.values()))!=len(m):raise ValueError('Nonbijective identity map')
 matrix=parse_matrix(text,row['code_map'],m)
 return {cid:[m[code] for code in values] for cid,values in matrix.items()}

def transform(row):
 if [m['role'] for m in row['messages']]!=['system','user']:raise ValueError('Unexpected original message schema')
 if any(k in row for k in ['response_code_map','response_patient_codes','response_representation']):raise ValueError('Already transformed')
 m=mapping(row);inverse={pid:code for code,pid in m.items()};new=copy.deepcopy(row)
 original_system=row['messages'][0]['content']
 if original_system.count(OLD_OUTPUT)!=1 or original_system.count(OLD_RELATIVE)!=1:raise ValueError('Unfrozen original instruction')
 new['messages'][0]['content']=original_system.replace(OLD_OUTPUT,NEW_OUTPUT).replace(OLD_RELATIVE,NEW_RELATIVE)+'\n'+CODE_INSTRUCTION
 table=[{'response_code':code,'original_patient_id':pid} for code,pid in m.items()]
 new['messages'][1]['content']=row['messages'][1]['content']+TABLE_PREFIX+json.dumps(table,ensure_ascii=False,separators=(',',':'))
 new['response_code_map']=m;new['response_patient_codes']=list(m);new['response_representation']=REPRESENTATION
 if 'answer' in row:
  answer=parse_matrix(row['answer'],row['code_map'],row['patient_ids'])
  if json.dumps(answer,ensure_ascii=False,separators=(',',':'))!=row['answer']:raise ValueError('Original answer is not the frozen compact serialization')
  new['answer']=json.dumps({cid:[inverse[pid] for pid in values] for cid,values in answer.items()},ensure_ascii=False,separators=(',',':'))
  if json.dumps(decode_response(new['answer'],new),ensure_ascii=False,separators=(',',':'))!=row['answer']:raise ValueError('Target inverse failed')
 return new
