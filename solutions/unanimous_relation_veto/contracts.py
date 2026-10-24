"""Synthetic distinguishing contracts; never reads task predictions or gold."""
import copy,json
from run import stream_evidence,veto_keys,edit_lines,all_pairs,classify,STREAMS

def main():
 patient={'d':{'P1','P2'}}
 def prompt(rid='r',values=None,pids=None):return dict(row_id=rid,pmc_id='d',code_map=values or {'C0':'HP:0000001'},patient_ids=pids or ['P1','P2'])
 def canon(row,mapping=None):return dict(row_id=row['row_id'],pmc_id='d',candidate_patients=mapping if mapping is not None else {c:[] for c in row['code_map']},strict_complete=True,abstention=None)
 def raw(row):return dict(row_id=row['row_id'],pmc_id='d',natural_eos=True,hit_token_limit=False)
 key=('d','P1','HP:0000001');q=prompt();c=canon(q);r=raw(q);neg=stream_evidence([q],[c],[r],patient)
 def removed(evidence,ground=set()):return veto_keys({key},{s:evidence.get(s,neg) for s in STREAMS},ground)[0]
 names=[]
 def check(name,condition):assert condition,name;names.append(name)
 check('four_explicit_negative_removes',removed({})=={key})
 for s in STREAMS:
  check(s+'_missing_stream_coverage_keeps',not removed({s:{}}))
  positive=stream_evidence([q],[canon(q,{'C0':['P1']})],[r],patient)
  check(s+'_positive_keeps',not removed({s:positive}))
 check('grounded_positive_protects',not removed({}, {key}))
 check('grounded_absence_is_not_negative_by_itself',not removed({'Q':{}},set()))
 for field,value in [('natural_eos',False),('hit_token_limit',True)]:
  bad=dict(r,**{field:value});e=stream_evidence([q],[c],[bad],patient);check(field+'_unknown_keeps',not removed({'C':e}))
 for bad in [dict(c,strict_complete=False,abstention='parsefail',candidate_patients={}),dict(c,candidate_patients={})]:
  check('abstention_or_missing_CID_keeps',not removed({'Q':stream_evidence([q],[bad],[r],patient)}))
 check('missing_row_metadata_keeps',not removed({'Q':stream_evidence([q],[c],[],patient)}))
 pp=prompt(pids=['P2']);check('missing_patient_keeps',not removed({'Q':stream_evidence([pp],[canon(pp)],[raw(pp)],patient)}))
 pp=prompt(values={'C0':'HP:0000002'});check('missing_candidate_keeps',not removed({'Q':stream_evidence([pp],[canon(pp)],[raw(pp)],patient)}))
 pp=prompt(values={'C0':'HP:0000001;HP:0000002'})
 compound=stream_evidence([pp],[canon(pp)],[raw(pp)],patient);check('compound_negative_never_expands',classify(compound[key])=='unknown' and not removed({'Q':compound}))
 compound=stream_evidence([pp],[canon(pp,{'C0':['P1']})],[raw(pp)],patient);check('compound_positive_atom_protects',classify(compound[key])=='positive' and not removed({'Q':compound}))
 q2=prompt('r2');positive=canon(q2,{'C0':['P1']})
 e=stream_evidence([q,q2],[c,positive],[r,raw(q2)],patient);check('duplicate_coverage_conflicting_positive_keeps',not removed({'S1':e}))
 bad=canon(q2);bad.update(strict_complete=False,abstention='invalid',candidate_patients={})
 e=stream_evidence([q,q2],[c,bad],[r,raw(q2)],patient);check('one_valid_one_abstain_keeps',not removed({'S1':e}))
 pp=prompt('compound',{'C0':'HP:0000001;HP:0000002'})
 e=stream_evidence([q,pp],[c,canon(pp)],[r,raw(pp)],patient);check('atomic_negative_plus_compound_unknown_keeps',not removed({'S1':e}))
 pp=prompt(values={'C0':'weakness; fatigue'});e=stream_evidence([pp],[canon(pp)],[raw(pp)],patient);nk=('d','P1','weakness; fatigue');check('NIL_semicolon_literal_atomic',classify(e[nk])=='negative' and ('d','P1','weakness') not in e)
 pp=prompt(values={'C0':'HP:0000001;weakness'});e=stream_evidence([pp],[canon(pp)],[raw(pp)],patient);check('mixed_malformed_HP_not_negative',classify(e[('d','P1','HP:0000001;weakness')])=='unknown')
 doc={'pmc_id':'d','entities':[{'text':'x'}],'association':[{'patient_id':'P1','phenotype':['HP:0000003;HP:0000001;HP:0000002','weakness; fatigue'],'extra':7},{'patient_id':'P2','phenotype':['HP:0000001']}],'other':'literal'}
 line=json.dumps(doc,ensure_ascii=False)+'\n';changed=edit_lines([line],{key});result=json.loads(changed[0])
 check('compound_partial_removal_order',result['association'][0]['phenotype']==['HP:0000003;HP:0000002','weakness; fatigue'])
 check('compound_other_patient_unchanged',result['association'][1]==doc['association'][1])
 check('association_extra_fields_unchanged',result['association'][0]['extra']==7)
 check('pairset_exact_difference',all_pairs([result])==all_pairs([doc])-{key})
 check('no_deletion_byteexact',edit_lines([line],set())==[line])
 check('reordered_rows_keyed_not_zipped',stream_evidence([q,q2],[canon(q2),c],[raw(q2),r],patient)==stream_evidence([q,q2],[c,canon(q2)],[r,raw(q2)],patient))
 return dict(status='PASS_SYNTHETIC_DISTINGUISHING_VETO_CONTRACTS',checks=len(names),names=names,task_predictions_read=False,gold_read=False,model_calls=0,fit_calls=0,quality_calls=0)
if __name__=='__main__':print(json.dumps(main(),indent=2))
