"""Independent set-valued response policy; original text and codec stay untouched."""
import json

def decode_set(text,row,original,*,natural_eos,hit_token_limit):
    if not natural_eos or hit_token_limit:
        raise ValueError('No natural EOS or generation reached cap')
    m=row['response_code_map']
    # Exact original identity-map contract, before reading response values.
    if list(m)!=row['response_patient_codes'] or list(m.values())!=row['patient_ids'] or len(set(m.values()))!=len(m):
        raise ValueError('Nonbijective identity map')
    matrix=json.loads(text,object_pairs_hook=original.unique_object)
    if not isinstance(matrix,dict) or set(matrix)!=set(row['code_map']):
        raise ValueError('Require every and only candidate CID')
    duplicates=[];result={}
    for cid,values in matrix.items():
        if not isinstance(values,list) or any(type(x) is not str for x in values):
            raise ValueError('Require list of string patient codes')
        if set(values)-set(m):raise ValueError('Unknown patient response code')
        first={};unique=[]
        for index,value in enumerate(values):
            if value in first:
                duplicates.append(dict(cid=cid,response_patient_code=value,removed_index=index,kept_index=first[value]))
            else:first[value]=index;unique.append(value)
        result[cid]=[m[value] for value in unique]
    return result,duplicates
