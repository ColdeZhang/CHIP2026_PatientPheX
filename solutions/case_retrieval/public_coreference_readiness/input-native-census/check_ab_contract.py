"""Deterministic synthetic adapter checks only; reuses four frozen native outputs."""
import pathlib,sys,json,copy,hashlib,collections
P=pathlib.Path(__file__).parent;R=P.parent;sys.path.insert(0,str(R))
from anchor_mapping import text_view,original_anchors
from anchor_mapping_ab import map_clusters_ab,inject_ab
fixture=json.loads((R/'synthetic-offset-contract.json').read_text());native=json.loads((R/'attempt2/synthetic-native-clusters.json').read_text());rows=[]
for x,n in zip(fixture['examples'],native['rows']):
 assert x['id']==n['fixture_id']
 doc={'pmc_id':'synthetic:'+x['id'],'full_text':[x['passage']],'patient':[{'patient_id':a['patient_id'],'mention':copy.deepcopy(a['mention'])} for a in x['patient']]}
 text,segments=text_view(doc);result={'fixture':x['id']}
 for mode in ['A','B']:
  m=map_clusters_ab(doc,text,segments,n['clusters_local_char_offsets'],{'synthetic_native_sha256':hashlib.sha256((R/'attempt2/synthetic-native-clusters.json').read_bytes()).hexdigest()},mode);derived,side=inject_ab(doc,m)
  result[mode]={'statuses':dict(collections.Counter(c['status'] for c in m['clusters'])),'added_mentions':side['added_mentions']}
 # Using the same text/anchors, a single NP covering all patients MUST reject whole cluster.
 anchors=original_anchors(doc);starts=[s for s,e in anchors];ends=[e for s,e in anchors];base=x['passage']['offset'];joined=[min(starts)-base,max(ends)-base]
 for mode in ['B']:
  m=map_clusters_ab(doc,text,segments,[[joined]],{'synthetic_adversarial':'one NP covers both given patients'},mode);assert m['clusters'][0]['status']=='conflicting_patients';derived,side=inject_ab(doc,m);assert derived==doc and not side['added_mentions']
 result['multi_patient_NP_rejected']=True
 # Same given original coordinate assigned two IDs is not a unique identity.
 shared=copy.deepcopy(doc);shared['patient'][1]['mention'].append(copy.deepcopy(shared['patient'][0]['mention'][0]));s=shared['patient'][0]['mention'][0];local=[s['offset']-base,s['offset']-base+s['length']]
 m=map_clusters_ab(shared,text,segments,[[local]],{'synthetic_adversarial':'shared given coordinate'},'B');assert m['clusters'][0]['status']=='conflicting_patients';assert not inject_ab(shared,m)[1]['added_mentions'];result['shared_coordinate_rejected']=True
 # Invalid cross-passage NP rejects even if an anchor is otherwise present.
 multi=copy.deepcopy(doc);multi['full_text'].append({'offset':base+len(text)+100,'text':'She remains well.'});v,segs=text_view(multi)
 m=map_clusters_ab(multi,v,segs,[[[0,len(v)]]],{'synthetic_adversarial':'cross passage'},'B');assert m['clusters'][0]['status']=='invalid_span';assert not inject_ab(multi,m)[1]['added_mentions'];result['cross_passage_rejected']=True
 rows.append(result)
out={'status':'PASS_adapter_contract_only','frozen_before_any_corpus_native':True,'rows':rows,'A_unique_clusters':sum(r['A']['statuses'].get('unique_patient',0) for r in rows),'B_unique_clusters':sum(r['B']['statuses'].get('unique_patient',0) for r in rows),'B_added_mentions':sum(len(r['B']['added_mentions']) for r in rows),'limitation':'A/B cannot prove patient identity or clinical attribution. NP with one supplied anchor and an unanchored other person can still attach; no unsupported semantic safeguard claimed.'}
(P/'ab-synthetic-contract-result.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
