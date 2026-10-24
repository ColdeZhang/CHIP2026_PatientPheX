"""Distinguishing toy contracts; never read actual predictions or gold."""
import json,sys
from pathlib import Path
import run
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts')
import evaluate as ev
def main():
 checks={}
 def ck(name,value):assert value,name;checks[name]=True
 key=('d','p','HP:0000001');other=('d','p','HP:0000002');src={s:set() for s in 'GSQC'}
 dc={'d':{('hpo','HP:0000001')}}
 ck('document_support_alone_keeps',run.select({key},dc,src)[0]=={key})
 ck('unsupported_zero_votes_deletes',not run.select({key},{},src)[0])
 for s in src:
  x={k:set(v) for k,v in src.items()};x[s].add(key);ck('one_pure_'+s+'_insufficient',not run.select({key},{},x)[0])
 for a,b in [('G','S'),('G','Q'),('G','C'),('S','Q'),('S','C'),('Q','C')]:
  x={s:({key} if s in (a,b) else set()) for s in src};ck('two_'+a+b+'_keeps',run.select({key},{},x)[0]=={key})
 ck('patient_support_not_reassigned',not run.select({('d','p2',key[2])},{},{s:{key} for s in src})[0])
 ck('document_support_shared_all_patients',run.select({key,('d','p2',key[2])},dc,src)[0]=={key,('d','p2',key[2])})
 ck('support_from_other_document_rejected',not run.select({('e','p',key[2])},dc,src)[0])
 ck('votes_never_add_absent_relation',run.select(set(),dc,{s:{key} for s in src})[0]==set())
 task=dict(entities=[dict(identifier='HP:0000001;HP:0000002',text='compound'),dict(identifier='HP:0000003',text='no',note='NO'),dict(identifier='-1',text='Literal; NIL'),dict(identifier='-1',text='Case'),dict(identifier='MESH:D000001',text='mesh')])
 terms={'d':ev.document_concepts(task,task,'all','text')}
 ck('valid_compound_entity_supports_each_atom',run.doc_supported(key,terms) and run.doc_supported(other,terms))
 ck('NO_entity_excluded',not run.doc_supported(('d','p','HP:0000003'),terms))
 ck('NIL_exact_semicolon',run.doc_supported(('d','p','Literal; NIL'),terms))
 ck('NIL_no_case_or_whitespace_or_substring',all(not run.doc_supported(('d','p',x),terms) for x in ['case','Literal','Literal; NIL ']))
 ck('nonHPO_identifier_not_concept_support',not run.doc_supported(('d','p','MESH:D000001'),terms))
 ck('no_ontology_or_alias_expansion',not run.doc_supported(('d','p','HP:0000004'),terms))
 ck('NIL_minus_one_not_text',not run.doc_supported(('d','p','-1'),terms))
 toy=dict(pmc_id='d',untouched={'x':[2]},entities=task['entities'],association=[dict(patient_id='p',other=[7],phenotype=['HP:0000002;HP:0000001;HP:0000003','Literal; NIL']),dict(patient_id='p2',phenotype=['HP:0000003'])])
 line=json.dumps(toy,ensure_ascii=False,indent=None)+'\n';kept={key,other,('d','p','Literal; NIL')};pl=run.edit_lines([line],kept);new=json.loads(pl[0])
 ck('compound_partial_order_preserved',new['association'][0]['phenotype']==['HP:0000002;HP:0000001','Literal; NIL'])
 ck('other_patient_can_empty_without_deletion',new['association'][1]==dict(patient_id='p2',phenotype=[]))
 ck('other_fields_and_bytes_preserved',run.nonrelation_bytes([line])==run.nonrelation_bytes(pl) and new['association'][0]['other']==[7])
 ck('exact_set_filter',run.pairs([new])==kept)
 ck('zero_deletion_byteexact',run.edit_lines([line],run.pairs([toy]))==[line])
 ck('malformed_mixed_relations_not_split',run.atoms('HP:0000001;NIL')==['HP:0000001;NIL'])
 def diag(stream,relations):return dict(stream=stream,row_id='r',pmc_id='d',strict_complete=True,actual_natural_eos=True,fresh=True,actual_hit_token_limit=False,actual_has_think_marker=False,whole_row_abstention=None,parser_error=None,relations=relations)
 ds=[diag('S1',[list(key)]),diag('S2',[])];s,counts=run.sample_intersection(ds,{'d':{'p'}},1)
 ck('S_is_actual_S1_intersect_S2',not s and counts=={'S1':1,'S2':0})
 ds[1]['relations']=[list(key)];s,_=run.sample_intersection(ds,{'d':{'p'}},1)
 ck('two_sampling_seeds_are_one_source',not run.select({key},{},dict(src,S=s))[0])
 ck('baseline_membership_is_not_sample_vote',not run.select({key},{},dict(src,G={key},S=set()))[0])
 ds=[diag('G',[list(other)]),*ds];s,_=run.sample_intersection(ds,{'d':{'p'}},1)
 ck('historical_greedy_diagnostics_not_grounded',s=={key})
 for field,value in [('strict_complete',False),('actual_natural_eos',False),('actual_hit_token_limit',True),('whole_row_abstention','failure')]:
  bad=[diag('S1',[]),diag('S2',[])];bad[0][field]=value
  try:run.sample_intersection(bad,{'d':{'p'}},1)
  except AssertionError:ck('incomplete_sample_'+field+'_fails',True)
  else:raise AssertionError('incomplete sample accepted')
 out=dict(status='PASS',checks=checks,count=len(checks),actual_prediction_read=False,gold_read=False,model_calls=0,fit_calls=0)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
