"""Stdlib fixtures only; never open any actual prediction, input or gold."""
from pathlib import Path
import ast
import copy
import importlib.util
import json
import sys
import tempfile
import time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def rejects(call):
    try:call()
    except (AssertionError,ValueError):return True
    raise AssertionError('Invalid synthetic case accepted')
def main():
    began=time.monotonic();control=load('four_source_control_fixture',ROOT/'control.py');helper=load('four_source_helpers_fixture',ROOT/'literal_union_helpers.py')
    concept=['HP:0000003','HP:0000001;HP:0000002','NIL;literal','HP:0000004']
    template=' {"pmc_id":"A", "entities":[{"text":"中文"}], "note":{"association":"nested"}, "n":1.00, "association" : [{"patient_id":"P1","phenotype":[]},{"patient_id":"P2","phenotype":[]}]} \r\n'
    base=[json.loads(template)]
    def prediction(p1=(),p2=()):
        value=copy.deepcopy(base)
        for item,values in zip(value[0]['association'],[p1,p2]):item['phenotype']=list(values)
        return value
    old=prediction([concept[0]]);reverse=prediction([concept[1]]);patient=prediction([], [concept[2]]);full=prediction([concept[3]])
    triple=prediction([concept[1],concept[0]],[concept[2]])
    components=dict(old=old,reverse=reverse,patient=patient,fulltext=full,triple=triple)
    original_template=template.replace('"phenotype":[]','"phenotype":'+json.dumps(triple[0]['association'][0]['phenotype']),1)
    original_template=original_template.replace('"phenotype":[]','"phenotype":'+json.dumps(triple[0]['association'][1]['phenotype']),1)
    prompts=[dict(row_id='A:0',pmc_id='A',patient_ids=['P1','P2'],code_map={str(i):v for i,v in enumerate(concept)},messages=[dict(role='system',content='fixture'),dict(role='user',content='fixture')])]
    inputs=[dict(pmc_id='A',patient=[dict(patient_id='P1'),dict(patient_id='P2')],full_text=[])]
    expected=dict(rows=1,documents=1,patients=2,entities=1,candidates=4,pairs=8)
    pool,pids=control.pool_from_prompts(prompts,inputs,expected)
    control.validate_components(components,inputs,pool,pids,expected)
    output,contract=control.compose(components,prompts,[original_template],helper)
    primary=[json.loads(output[control.ARMS[0]].decode())];diagnostic=[json.loads(output[control.ARMS[1]].decode())]
    assert control.literal_map(primary)['A']['P1']=={concept[0],concept[1],concept[3]}
    assert control.literal_map(diagnostic)['A']['P1']=={concept[1],concept[3]}
    assert control.literal_map(primary)['A']['P2']=={concept[2]}
    assert 'HP:0000001' not in control.literal_map(primary)['A']['P1']
    assert helper.nonrelation_bytes([original_template])==helper.nonrelation_bytes(output[control.ARMS[0]].decode().splitlines(keepends=True))==helper.nonrelation_bytes(output[control.ARMS[1]].decode().splitlines(keepends=True))
    assert b'"n":1.00' in output[control.ARMS[0]] and output[control.ARMS[0]].endswith(b' \r\n')
    missing=copy.deepcopy(components);missing['fulltext'][0]['association'].pop()
    duplicate=copy.deepcopy(components);duplicate['patient'][0]['association'][1]['patient_id']='P1'
    foreign=copy.deepcopy(components);foreign['reverse'][0]['association'][0]['phenotype'].append('HP:9999999')
    drift=copy.deepcopy(components);drift['fulltext'][0]['entities'][0]['text']='changed'
    wrong_triple=copy.deepcopy(components);wrong_triple['triple'][0]['association'][0]['phenotype']=[concept[1]]
    duplicate_rows=prompts+copy.deepcopy(prompts)
    tests=dict(literal_compounds_not_split=True,NIL_literal_preserved=True,primary_and_diagnostic_identities_exact=True,
        nonrelation_UTF8_CRLF_numeric_spelling_preserved=True,missing_patient_rejected=rejects(lambda:control.validate_components(missing,inputs,pool,pids,expected)),
        duplicate_PID_rejected=rejects(lambda:control.validate_components(duplicate,inputs,pool,pids,expected)),
        foreign_pool_value_rejected=rejects(lambda:control.validate_components(foreign,inputs,pool,pids,expected)),
        nonrelation_drift_rejected=rejects(lambda:control.validate_components(drift,inputs,pool,pids,expected)),
        incorrect_triple_literal_set_rejected=rejects(lambda:control.compose(wrong_triple,prompts,[original_template],helper)),
        duplicate_prompt_row_rejected=rejects(lambda:control.pool_from_prompts(duplicate_rows,inputs,{**expected,'rows':2})),
        duplicate_top_level_association_rejected=rejects(lambda:helper.association_span('{"association":[],"association":[]}')))
    with tempfile.TemporaryDirectory(dir=ROOT,prefix='synthetic-only-') as temp:
        folder=Path(temp);plan=dict(outputs={});frozen=dict(status='ALL_FOUR_COMPLETE_PREDICTIONS_FROZEN_BEFORE_GOLD',plan={'id':'plan'},source={'id':'source'},predictions={},gold_read=False,new_quality_calls=0)
        for fold in control.FOLDS:
            plan['outputs'][fold]={};frozen['predictions'][fold]={}
            for arm in control.ARMS:
                path=folder/(fold+'.'+arm+'.jsonl');path.write_text('{"fixture":true}\n');plan['outputs'][fold][arm]=str(path);frozen['predictions'][fold][arm]=control.ref(path)
        missing_freeze=copy.deepcopy(frozen);del missing_freeze['predictions']['F0'][control.ARMS[1]]
        spec=dict(gold_metadata_only=dict(path=str(folder/'synthetic-only-gold.jsonl'),bytes=3,sha256='not-yet-existent'))
        touched=[]
        def reader(path):touched.append(path);return [{'synthetic':True}]
        tests['missing_fourth_prediction_blocks_before_any_gold_access']=rejects(lambda:control.guarded_gold_load(spec,missing_freeze,plan,{'id':'plan'},{'id':'source'},reader))
        assert not touched and not Path(spec['gold_metadata_only']['path']).exists()
        gold=Path(spec['gold_metadata_only']['path']);gold.write_text('{}\n');spec['gold_metadata_only']=control.ref(gold)
        assert control.guarded_gold_load(spec,frozen,plan,{'id':'plan'},{'id':'source'},reader)==[{'synthetic':True}] and len(touched)==1
        broken=Path(frozen['predictions']['F0'][control.ARMS[1]]['path']);broken.write_text('{"fixture":false}\n')
        touched.clear();tests['fourth_SHA_drift_blocks_before_gold_reader']=rejects(lambda:control.guarded_gold_load(spec,frozen,plan,{'id':'plan'},{'id':'source'},reader));assert not touched
        original=folder/'exclusive';control.publish(original,b'first')
        try:control.publish(original,b'second')
        except FileExistsError:
            assert original.read_bytes()==b'first';tests['global_marker_exclusive']=True
        else:raise AssertionError('Exclusive marker overwritten')
    baseline=dict(Score=.5,macF1=.5)
    tests['Score_must_strictly_rise']=not control.fold_gate(dict(Score=.5,macF1=.6),baseline)
    tests['macro_must_not_fall']=not control.fold_gate(dict(Score=.6,macF1=.49),baseline)
    tests['macro_equality_allowed']=control.fold_gate(dict(Score=.6,macF1=.5),baseline)
    def result(primary,diagnostic):
        return {fold:{control.ARMS[0]:dict(fold_pass=True,delta_vs_triple=dict(Score=primary[i])),control.ARMS[1]:dict(fold_pass=True,delta_vs_triple=dict(Score=diagnostic[i]))} for i,fold in enumerate(control.FOLDS)}
    tie=result([.125,.25],[.25,.125]);tests['exact_twofold_tie_keeps_primary']=control.select(tie)['selected_candidate']==control.ARMS[0]
    stronger=result([.125,.125],[.25,.25]);tests['both_pass_higher_sum_can_choose_diagnostic']=control.select(stronger)['selected_candidate']==control.ARMS[1]
    stronger['F0'][control.ARMS[1]]['fold_pass']=False;tests['one_failed_fold_rejects_whole_arm']=control.select(stronger)['selected_candidate']==control.ARMS[0]
    stronger['F1'][control.ARMS[0]]['fold_pass']=False;tests['both_fail_selects_none']=control.select(stronger)['selected_candidate'] is None
    tree=ast.parse((ROOT/'control.py').read_text());main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    calls=[n for n in ast.walk(main_node) if isinstance(n,ast.Call)]
    freeze_line=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='save' and 'common-freeze.json' in ast.unparse(n.args[0]))
    eval_import=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='load_module' and n.args[0].value=='evaluate')
    validator_import=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='load_module' and n.args[0].value=='validate_predictions')
    gold_line=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='guarded_gold_load')
    score_line=next(n.lineno for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=='evaluate_report')
    tests['AST_freeze_H50_gold_score_order']=freeze_line<validator_import<eval_import<gold_line<score_line
    assert all(tests.values()) and time.monotonic()-began<60
    result=dict(status='PASS_SYNTHETIC_ONLY_LITERAL_UNION_AND_FOURWAY_FREEZE_ORDER_CONTRACT',tests=tests,source=control.ref(ROOT/'control.py'),helpers=control.ref(ROOT/'literal_union_helpers.py'),fixture_source=control.ref(__file__),actual_predictions_or_inputs_opened=False,actual_gold_opened=False,evaluator_calls=0,model_GPU_remote=False,seconds=time.monotonic()-began)
    control.save(ROOT/'synthetic-result.json',result);print(json.dumps(result))
if __name__=='__main__':main()
