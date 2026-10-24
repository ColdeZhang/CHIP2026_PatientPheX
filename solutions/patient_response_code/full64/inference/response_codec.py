"""Strict unchanged PT-to-clinical-PID decoder; no repair or target transformation."""
import json

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

def decode_response(text,row):
 m=row['response_code_map']
 if list(m)!=row['response_patient_codes'] or list(m.values())!=row['patient_ids'] or len(set(m.values()))!=len(m):raise ValueError('Nonbijective identity map')
 matrix=parse_matrix(text,row['code_map'],m)
 return {cid:[m[code] for code in values] for cid,values in matrix.items()}

