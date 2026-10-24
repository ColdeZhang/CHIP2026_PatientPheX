"""Synthetic exact-offset adapter counterexamples, not model/competition quality tests."""
import json,pathlib
from anchor_mapping import original_anchors,text_view,map_clusters,inject_mentions,canonical_sha
P=pathlib.Path(__file__).parent;fixtures=json.loads((P/'synthetic-offset-contract.json').read_text());checked=[]
for x in fixtures['examples']:
 doc={'pmc_id':x['id'],'full_text':[x['passage']],'patient':[{'patient_id':p['patient_id'],'mention':[{k:m[k] for k in ['text','offset','length']} for m in p['mention']]} for p in x['patient']]};before=canonical_sha(doc);text,segs=text_view(doc)
 clusters=[[(m['local_start'],m['local_end']) for m in g['mentions']] for g in x['hand_constructed_identity_contract']]
 mapped=map_clusters(doc,text,segs,clusters,{'type':'synthetic_expected_cluster'});new,sidecar=inject_mentions(doc,mapped);assert canonical_sha(doc)==before
 for oldp,newp in zip(doc['patient'],new['patient']):assert newp['mention'][:len(oldp['mention'])]==oldp['mention']
 assert new['full_text']==doc['full_text'];original_anchors(new)
 mixed=map_clusters(doc,text,segs,[clusters[0]+clusters[1]],{'type':'synthetic_wrong_merge'});assert mixed['clusters'][0]['status']=='conflicting_patients';unchanged,empty=inject_mentions(doc,mixed);assert unchanged==doc and not empty['added_mentions']
 bad=map_clusters(doc,text,segs,[[(0,len(text)+1)]],{});assert bad['clusters'][0]['status']=='invalid_span'
 checked.append({'id':x['id'],'original_input_unchanged':True,'conflicting_patient_cluster_rejected':True,'invalid_span_rejected':True,'added_by_synthetic_cluster_only':len(sidecar['added_mentions'])})
# A discontinuous global passage map must not accept a span through the separator.
doc={'pmc_id':'gaps','patient':[],'full_text':[{'offset':50,'text':'Ada.'},{'offset':500,'text':'She.'}]};t,s=text_view(doc);bad=map_clusters(doc,t,s,[[(1,8)]],{});assert bad['clusters'][0]['status']=='invalid_span'
(P/'integration-contract-result.json').write_text(json.dumps({'status':'pass_synthetic_adapter_contract_only','fixtures':checked,'cross_passage_gap_rejected':True,'no_model_or_quality_evaluation':True},indent=2)+'\n');print('synthetic integration contract PASS')
