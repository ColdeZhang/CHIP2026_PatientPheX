"""Bind complete F0 supplies and both fixed assembly plans; never read gold."""
import argparse
import json
import os
from pathlib import Path
from common import ref,checked,read_json,universe,write_json,now,HPO
from common_f0 import candidates,literal_predictions,sampled_intersection


def main(runs, fusion_source):
    R=Path(runs);S=Path(__file__).resolve().parent;P=Path(fusion_source).resolve()
    L=R/'research/relation-calibration-20261024';F=R/'research/parallel-relation-fusion-F0-20261024';F.mkdir(exist_ok=True)
    sample=R/'research/nonthinking-sampled-consensus-F0-20261024';g=R/'research/grounded-generation-v1';q=R/'stage8-qwen3-relations'
    sp=json.loads((sample/'gpu-plan.json').read_text());forbidden=os.path.realpath(sp['quality_refs']['gold']['path'])
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            assert os.path.realpath(os.fsdecode(args[0]))!=forbidden, 'No F0 gold binding read/hash'
    __import__('sys').addaudithook(audit)
    refs={'sample_ancestry':ref(sample/'preparation-v2/complete.json'),'grounded_ancestry':ref(g/'F0-preparation-v1/ancestry.json'),'qwen_ancestry':ref(q/'f0-readiness/ancestry.json'),'sample_freeze':ref(sample/'assembled-v1/prediction-freeze.json'),'grounded_freeze':ref(g/'F0-pure-relations-v1/freeze.json'),'qwen_freeze':ref(q/'f0-v1/quality/complete-freeze.json'),'grounded_union_freeze':ref(g/'strong-support-union-F0-v1/freeze.json'),'F1_fit_ancestry':ref(L/'preparation-v1/ancestry.json'),'F1_fit_result':ref(L/'fit-v1/result.json')}
    a={k:read_json(v) for k,v in refs.items()}
    inp=sp['quality_refs']['input'];u=universe(inp,16,47);f0=set(u);t32=set(a['F1_fit_ancestry']['training_document_ids'])
    sa=a['sample_ancestry'];ga=a['grounded_ancestry'];qa=a['qwen_ancestry']
    assert set(sa['F0_ids'])==set(ga['F0_document_ids'])==f0
    assert set(sa['generation_train32'])==set(ga['train32_document_ids'])==t32 and len(t32)==32 and not t32&f0
    assert not sa['F0_intersection_all_task_fitted_ancestors'] and not qa['training_F0_overlap'] and not qa['F0_candidate_F0_overlap']
    assert set(sa['held_candidate_and_gate_train48'])==set(qa['F0_candidate_training_ids']) and len(sa['held_candidate_and_gate_train48'])==48 and not set(sa['held_candidate_and_gate_train48'])&f0
    assert qa['model_training_documents']==32 and qa['original_preparation']['sha256']=='356109e9b35d71e36c315b012831166419b4e007db061a289a04771abee69a47'
    assert qa['train124']['sha256']=='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
    for row in qa['training_candidate_generators']:assert not row['F0_overlap'] and row['generator_fold'] in (2,3) and row['target_fold'] in (2,3)
    assert ga['normalizer_train32_sha256']=='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    assert ga['normalizer_policy']['sha256']=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
    assert ga['public_bank_manifest']['sha256']=='443922dadf116ecc6d810af847b3ee9fbe513af9bd03f5896742543910a91f7e' and ga['train_only_aliases_and_no_held_quote_dictionary_updates']
    sf=a['sample_freeze'];gf=a['grounded_freeze'];qf=a['qwen_freeze'];guf=a['grounded_union_freeze']
    assert sf['documents']==gf['documents']==qf['documents']==16 and sf['patients']==gf['patients']==qf['patients']==47
    assert gf['all47_complete'] and gf['no_gold_no_score'] and qf['fresh_rows']==qf['strict_complete']==58
    assert qf['gold_reads_before_freeze']==qf['v2_calls_before_freeze']==0
    assert sf['sampled_fresh_rows']==116 and sf['gold_read'] is False and sf['quality_calls']==0
    raw={'grounded':gf['predictions']['new-only'],'qwen3':qf['predictions']['new-alone-unscored'],'sampled_diagnostics':sf['row_diagnostics']}
    unbound=json.loads((S/'F0-plan.template.json').read_text())
    unbound.update(status='BOUND_COMPLETE_F0_SOURCES',raw_sources=raw,baseline=sp['baseline'],sources={n:ref(S/n) for n in ('common.py','common_f0.py','assemble_F0.py')})
    _,rows,counts=candidates(unbound);base=literal_predictions(sp['baseline'],u)
    sources={'grounded':literal_predictions(raw['grounded'],u),'sampled':sampled_intersection(raw['sampled_diagnostics'],u,58)[0],'qwen3':literal_predictions(raw['qwen3'],u)}
    protected={'grounded':guf['prediction'],'sampled':next(iter(sf['endpoints'].values())),'qwen3':qf['predictions']['protected-union']}
    assert len(base)==315
    for name,r in protected.items():assert literal_predictions(r,u)==base|sources[name],name
    proof={'status':'PASS_ALL_COMPLETE_F0_RAW_SOURCES_AND_FIXED_ANCESTRY','utc':now(),'all_task_ancestors_exclude_F0':True,'references':refs,'raw_sources':raw,'protected_sources':protected,'counts':counts,'F0_document_ids':sorted(f0),'generation_train32_ids':sorted(t32),'candidate_train48_F0_overlap':[],'all_protected_sources_exact_B0_union_full_raw':True,'F1_source_and_model_unchanged':True,'fit_model':ref(L/'fit-v1/model.json'),'interface':'F0-only v2 unit compatibility: pure compound HPO split, other literal unchanged; complete explicitly empty sampler abstention accepted. No answer repair.','grounded_literal_relations':sum(not HPO.fullmatch(k[2]) for k in sources['grounded']),'source_outcome_not_used_to_select':True,'grounded_generation_limits':'47 complete windows, 44 EOS/3 cap, two strict JSON row abstentions; owner frozen outputs consumed unchanged.','crossfold_limits':'Sample/Qwen held candidates train48 here versus train32 in F1; all exclude F0. Public pretrained corpus membership unknown. F1 was explicit LR meta-training.','gold_read_or_hashed':False}
    proofref=write_json(L/'F0-ancestry.json',proof);unbound['F0_ancestry']=proofref
    write_json(S/'F0-plan.json',unbound)
    fixed={'scope':'train64_internal_F0_development_not_blind','output':str(F/'actual-v1'),'baseline_score':.7115075800266846,'expected':{'base_pairs':315,'entities':947,'patients':47,'source_additions':{k:len(v-base) for k,v in sources.items()}},'refs':{'source':ref(P/'run_f0.py'),'baseline':sp['baseline'],'baseline_metrics':sp['baseline_metrics'],**protected,**sp['quality_refs']},'ancestry':proofref,'joint_freeze_required':True,'single_fixed_endpoint':'B0 union at least two of three complete raw signals; same as voting over their B0 additions'}
    write_json(P/'plan-F0.json',fixed)
    print(json.dumps({'LR_plan':ref(S/'F0-plan.json'),'fixed_plan':ref(P/'plan-F0.json'),'ancestry':proofref,'counts':counts,'source_additions':fixed['expected']['source_additions'],'literal_grounded':proof['grounded_literal_relations']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--fusion-source',required=True);a=p.parse_args();main(a.runs,a.fusion_source)
