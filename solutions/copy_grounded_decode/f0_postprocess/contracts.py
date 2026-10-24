"""Distinct synthetic row-abstention checks; no actual model output or held gold."""
import json,sys
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S));sys.path.insert(0,'/workspace/patientphex-worktrees/explore-grounded-generation')
from terminal_contract import interpret_row
from solutions.grounded_generation.inference.decode import interpret
sys.path.insert(0,str(S.parent));from local_common import ref,save

def main():
 doc=dict(full_text=[dict(offset=10,text='café café')]);calls=[]
 def parser(raw,doc,eos,cap):calls.append((eos,cap));return interpret(raw,doc,eos,cap)
 cap=dict(response='[["U0000","café",0]]',actual_last_token_eos=False,hit_token_limit=True)
 r=interpret_row(cap,doc,dict(pass_strict=False),parser);assert r['status']=='abstain' and r['anchors']==[] and calls==[(False,True)]
 calls.clear();failed=dict(response='[["U0000","café",0]]',actual_last_token_eos=True,hit_token_limit=False)
 r=interpret_row(failed,doc,dict(pass_strict=False),parser);assert r['status']=='abstain' and r['anchors']==[] and not calls
 r=interpret_row(failed,doc,dict(pass_strict=True),parser);assert r['status']=='complete' and r['anchors'][0]['offset']==10 and r['rejected']==[]
 empty=dict(response='[]',actual_last_token_eos=True,hit_token_limit=False);r=interpret_row(empty,doc,dict(pass_strict=True),parser);assert r['status']=='complete' and r['anchors']==[]
 save(S/'evidence/whole-row-contract.json',dict(status='PASS',source=ref(__file__),terminal_contract=ref(S/'terminal_contract.py'),cap_no_prefix_salvage=True,EOS_strict_failure_no_parser_or_item_salvage=True,strict_success_original_parser=True,empty_array_valid=True,actual_raw_read=False,gold_read=False,quality_calls=0,model_calls=0))
if __name__=='__main__':main()
