"""Differentiating contracts for stable legal-value deduplication only."""
import copy
from codec import decode_set

def run(original):
    row=dict(code_map={'C000':'HP:1','C001':'HP:2'},response_code_map={'PT001':'P1','PT002':'O2','PT003':'P3'},response_patient_codes=['PT001','PT002','PT003'],patient_ids=['P1','O2','P3'])
    passed=[]
    def check(name,text,expected=None,folds=0,reject=False,eos=True,cap=False,metadata=None):
        try:value,duplicates=decode_set(text,metadata or row,original,natural_eos=eos,hit_token_limit=cap)
        except (ValueError,TypeError):
            assert reject,name;passed.append(name);return
        assert not reject and value==expected and len(duplicates)==folds,(name,value,duplicates);passed.append(name)
    simple='{"C000":["PT002","PT001"],"C001":[]}'
    check('original_valid_identical',simple,original.decode_response(simple,row))
    check('stable_order_nonadjacent_duplicates','{"C000":["PT002","PT001","PT002","PT001","PT003"],"C001":[]}',{'C000':['O2','P1','P3'],'C001':[]},2)
    check('no_cross_CID_collapse','{"C000":["PT001","PT001"],"C001":["PT001"]}',{'C000':['P1'],'C001':['P1']},1)
    check('escaped_equal_legal_value','{"C000":["PT001","\\u0050T001"],"C001":[]}',{'C000':['P1'],'C001':[]},1)
    bad={
        'duplicate_JSON_key':'{"C000":[],"C000":[],"C001":[]}',
        'escaped_duplicate_JSON_key':'{"C000":[],"\\u0043000":[],"C001":[]}',
        'unknown_code':'{"C000":["PT999"],"C001":[]}',
        'duplicate_unknown_code':'{"C000":["PT999","PT999"],"C001":[]}',
        'valid_duplicates_do_not_hide_unknown':'{"C000":["PT001","PT001","PT999"],"C001":[]}',
        'missing_CID':'{"C000":[]}',
        'extra_CID':'{"C000":[],"C001":[],"C002":[]}',
        'not_object':'[]',
        'not_list':'{"C000":"PT001","C001":[]}',
        'integer_value':'{"C000":[1],"C001":[]}',
        'boolean_value':'{"C000":[true],"C001":[]}',
        'null_value':'{"C000":[null],"C001":[]}',
        'unclosed':'{"C000":["PT001","PT001"],"C001":[]',
        'trailing_text':'{"C000":[],"C001":[]} garbage',
        'code_fence':'```json\n{"C000":[],"C001":[]}\n```'}
    for name,text in bad.items():check(name,text,reject=True)
    check('non_EOS',simple,reject=True,eos=False);check('cap_with_EOS_rejected',simple,reject=True,cap=True)
    check('non_EOS_cap',simple,reject=True,eos=False,cap=True)
    wrong=copy.deepcopy(row);wrong['response_code_map']['PT002']='P1';check('nonbijective_identity',simple,reject=True,metadata=wrong)
    wrong=copy.deepcopy(row);wrong['response_patient_codes'].reverse();check('map_order_changed',simple,reject=True,metadata=wrong)
    return dict(status='PASS_SET_DECODER_CONTRACTS',count=len(passed),checks=passed,original_codec_unchanged=True,NN_calls=0)
