"""Boundary cases for the new policy against the unchanged CanonicalList state."""
from types import SimpleNamespace
from prefix import trace

class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):
        return SimpleNamespace(ids=[200001] if text=='[\n' else [200002] if text=='[]' else [ord(x) for x in text])
    def decode(self,ids,skip_special_tokens=False):
        return ''.join({200001:'[\n',200002:'[]',200003:'<|im_end|>'}.get(i,chr(i) if i<200000 else '') for i in ids)

def run(CanonicalList):
    tk=CharacterTokenizer();names={'A':'HP:1','AB':'HP:2','Escaped "quote" \\ slash':'HP:3'};codec=CanonicalList(tk,names,200003)
    passed=[]
    def check(label,ids,wanted,eos=False,reject=False):
        raw=tk.decode(ids);body=raw[:-len('<|im_end|>')] if eos else raw
        try:r=trace(codec,ids,body,eos)
        except (ValueError,AssertionError):
            assert reject,label;passed.append(label);return
        assert not reject and r['hpo_ids']==wanted,(label,r);passed.append(label)
    full=codec.serialize_ids(['A','AB'])
    check('full_two_EOS',full+[200003],['HP:1','HP:2'],True)
    first=codec.open+codec.parts['A',True]
    check('stop_at_comma',first[:-1],['HP:1'])
    check('stop_after_closed_quote',codec.open+tk.encode('"A"').ids,['HP:1'])
    check('unclosed_name',codec.open+tk.encode('"A').ids,[])
    check('middle_second_name',first+tk.encode('"A').ids,['HP:1'])
    escaped=codec.open+codec.parts['Escaped "quote" \\ slash',False]
    check('escaped_complete',escaped+[200003],['HP:3'],True)
    check('partial_escape',codec.open+tk.encode('"Escaped \\').ids,[])
    check('complete_list_without_EOS',codec.serialize_ids(['A']),['HP:1'])
    check('empty_EOS',codec.empty+[200003],[],True)
    check('unknown_rejected',codec.open+tk.encode('"Unknown"\n]').ids,[],reject=True)
    check('duplicate_rejected',first+codec.parts['A',False],[],reject=True)
    check('EOS_inside_name_rejected',codec.open+tk.encode('"A').ids+[200003],[],True,True)
    check('extra_after_list_rejected',full+[65],[],reject=True)
    check('trailing_after_EOS_rejected',full+[200003,65],[],reject=True)
    return dict(status='PASS_NEW_PREFIX_BOUNDARIES',checks=passed,count=len(passed),no_model=True)
