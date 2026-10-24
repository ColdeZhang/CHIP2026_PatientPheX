"""Discriminating CPU contracts; synthetic values only, no model or quality."""
import copy
import sys
from pathlib import Path
import numpy as np
from common import deny_gold, module, primary, read, ref, save
from adapted_semantic import admitted, alias_index, legacy_modules, load_policy, validate_provenance

S=Path(__file__).resolve().parent


def rejected(fn):
    try:fn()
    except (AssertionError,ValueError):return
    raise AssertionError('Required rejection did not occur')


def main():
    deny_gold();e=read(S/'evidence/preparation.json');p=read(e['preparation']['path'])
    sources={k:p['sources'][k] for k in ['strict.py','semantic.py','public_bank.py']}
    strict,public,semantic=legacy_modules(sources)
    policy=load_policy(S/'policy_v1.json',p['original_policy']['path'])
    b=read(S/'bank-manifest.json')
    validate_provenance(b,b['checkpoint'],b['bank'],p['public_labels'],p['hpo']['sha256'])
    for key,value in [('no_task_fit',True),('task_adapted',False),('encoder_parameter_dtype','float16'),('old_public_encoder_certificate_reused',True)]:
        wrong=copy.deepcopy(b);wrong[key]=value
        rejected(lambda:validate_provenance(wrong,b['checkpoint'],b['bank'],p['public_labels'],p['hpo']['sha256']))
    legacy=copy.deepcopy(b);legacy['checkpoint']['sha256']='0'*64
    rejected(lambda:validate_provenance(legacy,legacy['checkpoint'],legacy['bank'],p['public_labels'],p['hpo']['sha256']))
    base={'quote':'progressive weakness','status':'abstain','reason':'unmatched','ids':[],'identifier':None,'match':None}
    def score(top=.90,second=.85,n=6):
        return dict(quote=base['quote'],token_count=n,candidates=[dict(identifier='HP:0001250',score=top,alias='progressive weakness'),dict(identifier='HP:0001251',score=second,alias='other weakness')])
    assert semantic.decide(base,score(),policy)['status']=='mapped'
    assert semantic.decide(base,score(.8999),policy)['reason']=='semantic_cosine_below_fixed_boundary'
    assert semantic.decide(base,score(.92,.90),policy)['reason']=='semantic_distinct_concept_margin_too_small'
    assert semantic.decide(base,score(n=33),policy)['reason']=='semantic_query_would_truncate'
    assert semantic.decide(base,score(.95,.95),policy)['status']=='abstain'
    rejected(lambda:semantic.decide(base,None,policy))
    protected=[dict(base,status='mapped',reason='unanimous_whole_span_alias',identifier='HP:0001250',ids=['HP:0001250'],match='literal'),
               dict(base,quote='rare syndrome;text',status='nil',reason='unanimous_whole_span_alias',identifier='-1',ids=['-1'],match='literal'),
               dict(base,reason='known_alias_ambiguous')]
    for row in protected:
        actual=semantic.decide(row,None,policy)
        assert all(actual[k]==v for k,v in row.items())
    assert semantic.precheck('ABCD',policy)=='semantic_unknown_abbreviation'
    # Shared public alias must produce tied DISTINCT concepts, not a false alias-rank margin.
    rows=[dict(text='shared',identifiers=['HP:0001250','HP:0001251']),dict(text='other',identifiers=['HP:0001252'])]
    class Fixture:pass
    bank=Fixture();bank.rows=rows;bank.bank=np.zeros((2,768),dtype=np.float32);bank.bank[0,0]=1;bank.bank[1,1]=1
    bank.concepts,bank.by_concept,bank.alias_indices,bank.concept_indices=alias_index(rows)
    ranked=semantic.PublicConceptBank.rank(bank,bank.bank[0],top_k=3)
    assert [r['identifier'] for r in ranked]==['HP:0001250','HP:0001251','HP:0001252']
    assert [r['score'] for r in ranked]==[1.,1.,0.]
    # Original itemwise parser retains a valid item despite another invalid quote/duplicate.
    sys.path.insert(0,str(Path(p['decode']['path']).parents[3]))
    decode=module('_adapted_original_decode_contract',p['decode']);dc=decode.contracts()
    assert dc['partial_invalid_items_abstain'] and dc['EOS_and_cap_gates']
    assert primary({'b'},{'same_old_g'},set(),set(),set())=={'b'}
    assert primary({'b'},{'new'}, {'new'},set(),set())=={'b','new'}
    assert primary({'b'},set(),{'new'},{'new'},{'new'})=={'b'}
    sys.path.insert(0,str(Path(p['original_support']['path']).parent))
    support=module('_adapted_byte_support',p['original_support']);spans,mask=support.byte_helpers(p['byte_helper'])
    raw='{"pmc_id":"D", "entities":[],"association":[{"patient_id":"P", "phenotype":["HP:0001250;HP:0001251", "unmapped;text"], "extra": 7}]}\n'
    rendered=support.replace_phenotypes(raw,{'P':['HP:0001250;HP:0001251','unmapped;text','HP:0001252']},spans,mask)
    assert mask(raw)==mask(rendered)
    cfg=dict(execution_authorized=True,launch_valid_until_unix=100,budget_seconds=180,grace_seconds=30)
    admitted(cfg,99);rejected(lambda:admitted(cfg,101));rejected(lambda:admitted(dict(cfg,execution_authorized=False),99))
    save(S/'evidence/contracts.json',dict(status='PASS_CPU_SYNTHETIC_NO_MODEL_OR_GOLD',source=ref(__file__),
        adapted_source=ref(S/'adapted_semantic.py'),strict_exact_NIL_ambiguity_unchanged=True,old_cosine_and_margin_boundary=True,
        missing_eligible_score_rejected=True,over32_no_truncation=True,shared_alias_distinct_concept_tie=True,
        original_itemwise_parser=dc,no_old_G_self_support=True,byte_protection=True,
        old_public_certificate_and_FP16_weight_substitution_rejected=True,expired_or_unauthorized_launch_rejected=True,
        model_loads=0,forward_calls=0,quality_calls=0))
    print('PASS CPU synthetic contracts; no model/gold/quality')


if __name__=='__main__':main()
