"""B-only marker control: preserve selected old80 evidence byte-for-byte after tag removal."""
import ast,pathlib,json,hashlib,copy,collections,re,datetime,sys
P=pathlib.Path(__file__).parent;R=P.parent;C=R/'input-native-census';B=pathlib.Path('/workspace/patientphex-runs/round2-20261023/case_retrieval/qwen35-preparation/sft-meta1-fusion');sys.path[:0]=[str(R),str(C)]
from anchor_mapping import text_view,canonical_sha
from anchor_mapping_ab import map_clusters_ab,inject_ab
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ref(p):p=pathlib.Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def rows(p):return [json.loads(x) for x in pathlib.Path(p).read_text().splitlines()]
def dump(p,x):pathlib.Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
oldplan=json.loads((B/'public-new-relations/run-plan.json').read_text());refs={k:oldplan['references'][k] for k in ['input','old_entities','old_prompts','old_raw','old_manifest','parser_base']}
for x in refs.values():assert ref(x['path'])==x
helper=R.parent/'public-coreference-feasibility/local-snapshots/13-prepare_pair_rag.py';assert sha(helper)=='16345e096b462ae9ba2fc974bb60d14ff15333b09009b4add1f0a0e1bb23ed97'
# Execute only fixed pure helper functions; do not import or initialize training/runtime code.
ns={'defaultdict':collections.defaultdict,'re':re,'json':json}
for path,names in [(pathlib.Path(refs['parser_base']['path']),{'association_keys'}),(helper,{'mark_patients','grouped_concepts','evidence','patient_anchors'})]:
 tree=ast.parse(path.read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names];assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
NOTE='Inline [COREF P1: text] markers are automatic coreference suggestions; verify them against the original clinical text before assigning a phenotype.'
SEP='\n\nCandidate phenotypes:\n'
inputdocs={str(x['pmc_id']):x for x in rows(refs['input']['path'])};entities={str(x['pmc_id']):x for x in rows(refs['old_entities']['path'])};oldrows=rows(refs['old_prompts']['path']);native={x['pmc_id']:x for x in rows(C/'full48/documents.jsonl') if x['group']=='F1'};native_sha=sha(C/'full48/documents.jsonl');assert len(inputdocs)==len(native)==16 and len(oldrows)==80
assert native_sha=='db0f3932c20ccb330dc67cde0bca0513f6baac418702b5c09ddcfbbbbeab1b71'
additions={};sources={}
for pid,doc in inputdocs.items():
 n=native[pid];text,segments=text_view(doc);mapping=n['modes']['B']['mapping'];source=mapping['clusters'][0]['source'] if mapping['clusters'] else {'unused':True};fresh=map_clusters_ab(doc,text,segments,n['clusters_local_char_offsets'],source,'B');assert fresh==mapping;derived,side=inject_ab(doc,mapping);assert side==n['modes']['B']['sidecar'];additions[pid]=side['added_mentions'];sources[pid]=side

def evidence_trace(doc,es):
 candidates=[]
 for ei,entity in enumerate(es):
  for pi,passage in enumerate(doc['full_text']):
   offset=int(passage['offset']);local=int(entity['offset'])-offset
   if not 0<=local<=local+int(entity['length'])<=len(passage['text']):continue
   left=max(0,local-500);right=min(len(passage['text']),local+int(entity['length'])+500);raw=passage['text'][left:right];marked=ns['mark_patients'](raw,offset+left,doc);section=str(passage.get('section_type',''))+' '+str(passage.get('type',''));priority=int('[PATIENT' in marked)+int(bool(re.search(r'case|patient|result|clinical|method',section,re.I)))
   ev={'quote':entity['text'],'section':section,'context':marked};trace={'passage_index':pi,'global_start':offset+left,'global_end':offset+right,'raw_text':raw,'candidate_entity_index':ei,'candidate_entity_offset':int(entity['offset']),'priority':priority,'evidence':ev};candidates.append(trace);break
 chosen=[];seen=set()
 for x in sorted(candidates,key=lambda x:(-x['priority'],x['candidate_entity_offset'])):
  if x['evidence']['context'] not in seen:chosen.append(x);seen.add(x['evidence']['context'])
  if len(chosen)>=4:break
 assert [x['evidence'] for x in chosen]==ns['evidence'](doc,es,500)
 return chosen

def decorate(doc,trace,adds):
 raw=trace['raw_text'];start=trace['global_start'];old=trace['evidence']['context'];assert '[COREF ' not in old;groups=collections.defaultdict(set)
 for patient in doc['patient']:
  for m in patient['mention']:
   s=int(m['offset'])-start;e=s+int(m.get('length',len(m['text'])))
   if 0<=s<e<=len(raw):groups[(s,e)].add(str(patient['patient_id']))
 events=[];position=0;pieces=[]
 for (s,e),ids in sorted(groups.items()):
  if s<position:continue
  prefix='[PATIENT '+'/'.join(sorted(ids))+': ';events.append((s,e,len(prefix)+1));pieces.extend([raw[position:s],prefix,raw[s:e],']']);position=e
 pieces.append(raw[position:]);assert ''.join(pieces)==old
 proposals=[]
 for a in adds:
  m=a['mention'];s=m['offset']-start;e=s+m['length']
  if not 0<=s<e<=len(raw):continue
  assert raw[s:e]==m['text']
  assert all(e<=os or oe<=s for os,oe,_ in events)
  shift=sum(extra for os,oe,extra in events if oe<=s);ms=s+shift;me=e+shift;assert old[ms:me]==m['text'];proposals.append((ms,me,a))
 proposals.sort(key=lambda x:(x[0],x[1],x[2]['patient_id']));pieces=[];position=0;cursor=0;applied=[]
 for ms,me,a in proposals:
  assert position<=ms;plain=old[position:ms];pieces.append(plain);cursor+=len(plain);prefix='[COREF '+a['patient_id']+': ';text=old[ms:me];wrapped=prefix+text+']';pieces.append(wrapped);applied.append({'new_start':cursor,'new_end':cursor+len(wrapped),'prefix':prefix,'text':text,'suffix':']','old_context_start':ms,'old_context_end':me,'patient_id':a['patient_id'],'mention':a['mention'],'cluster_id':a['cluster_id'],'source':a['source']});cursor+=len(wrapped);position=me
 pieces.append(old[position:]);new=''.join(pieces);restored=new
 for a in reversed(applied):
  assert restored[a['new_start']:a['new_end']]==a['prefix']+a['text']+a['suffix'];restored=restored[:a['new_start']]+a['text']+restored[a['new_end']:]
 assert restored==old
 return new,applied
newrows=[];audit=[];counts=collections.Counter();used=set();docsused=set()
for old in oldrows:
 doc=inputdocs[str(old['pmc_id'])];assert not doc.get('entities') and not doc.get('association');assert old['patient_ids']==[str(x['patient_id']) for x in doc['patient']];new=copy.deepcopy(old);assert len(old['messages'])==2 and old['messages'][0]['role']=='system' and old['messages'][1]['role']=='user';prefix,rawitems=old['messages'][1]['content'].split(SEP,1);items=json.loads(rawitems);assert json.dumps(items,ensure_ascii=False)==rawitems
 assert prefix=='Patient identities and global anchors:\n'+ns['patient_anchors'](doc)
 groups=ns['grouped_concepts'](entities[str(doc['pmc_id'])]['entities']);newitems=copy.deepcopy(items);evidenceaudit=[]
 for item,ni in zip(items,newitems):
  identifier=old['code_map'][item['code']];traces=evidence_trace(doc,groups[identifier]);assert item['evidence']==[x['evidence'] for x in traces]
  for idx,(ev,trace) in enumerate(zip(ni['evidence'],traces)):
   modified,applied=decorate(doc,trace,additions[str(doc['pmc_id'])]);ev['context']=modified;counts['contexts']+=1;counts['contexts_with_new_markers']+=bool(applied);counts['marker_occurrences']+=len(applied)
   for x in applied:used.add((str(doc['pmc_id']),x['patient_id'],x['mention']['offset'],x['mention']['length']));docsused.add(str(doc['pmc_id']))
   evidenceaudit.append({'code':item['code'],'identifier':identifier,'evidence_index':idx,'passage_index':trace['passage_index'],'global_start':trace['global_start'],'global_end':trace['global_end'],'old_context_sha256':hashlib.sha256(trace['evidence']['context'].encode()).hexdigest(),'new_context_sha256':hashlib.sha256(modified.encode()).hexdigest(),'applied':applied})
 new['messages'][0]['content']=old['messages'][0]['content']+'\n'+NOTE;new['messages'][1]['content']=prefix+SEP+json.dumps(newitems,ensure_ascii=False)
 # Restore all serialized user bytes and the sole appended system sentence.
 restored=copy.deepcopy(new);restoreitems=json.loads(restored['messages'][1]['content'].split(SEP,1)[1]);ai=iter(evidenceaudit)
 for item in restoreitems:
  for ev in item['evidence']:
   a=next(ai);v=ev['context']
   for x in reversed(a['applied']):assert v[x['new_start']:x['new_end']]==x['prefix']+x['text']+x['suffix'];v=v[:x['new_start']]+x['text']+v[x['new_end']:]
   ev['context']=v
 restored['messages'][1]['content']=prefix+SEP+json.dumps(restoreitems,ensure_ascii=False);assert restored['messages'][0]['content'].endswith('\n'+NOTE);restored['messages'][0]['content']=restored['messages'][0]['content'][:-(len(NOTE)+1)];assert restored==old
 changed=any(x['applied'] for x in evidenceaudit);counts['rows']+=1;counts['rows_with_new_markers']+=changed;newrows.append(new);audit.append({'row_id':old['row_id'],'pmc_id':str(old['pmc_id']),'original_prompt_canonical_sha256':canonical_sha(old),'derived_prompt_canonical_sha256':canonical_sha(new),'restored_exact':True,'all_nonmessage_fields_exact':True,'global_anchor_prefix_exact':True,'evidence':evidenceaudit})
assert sum(len(d['entities']) for d in entities.values())==1189;assert len(newrows)==80 and sum(len(r['code_map']) for r in newrows)==587 and sum(len(r['code_map'])*len(r['patient_ids']) for r in newrows)==1557
(P/'prompts.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in newrows));(P/'prompt-sidecars.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in audit))
counts.update(unique_B_mentions_used=len(used),F1_B_mentions_available=sum(len(x) for x in additions.values()),docs_with_new_markers=len(docsused));summary={'status':'CPU_prompt_construction_reversible_exact_before_token_audit','counts':dict(counts),'fixed_system_appendix':NOTE,'all80_strip_to_original_exact':True,'old_evidence_selection_order_and_global_anchors_unchanged':True,'B_only':True,'original_entities':1189,'concepts':587,'patient_concept_pairs':1557,'references':dict(refs,original_prompt_helper=ref(helper),coref_native_output=ref(C/'full48/documents.jsonl'),coref_gpu_plan=ref(C/'gpu-plan.json'),coref_input_manifest=ref(C/'input-manifest.json'),coref_mapping_source=ref(C/'anchor_mapping_ab.py'),original_mapping_source=ref(R/'anchor_mapping.py')),'outputs':{n:ref(P/n) for n in ['prompts.jsonl','prompt-sidecars.jsonl']},'source':ref(__file__),'no_gold_or_new_NN_access':True};dump(P/'preparation.json',summary);print(json.dumps({'counts':dict(counts),'prompts':summary['outputs']['prompts.jsonl'],'reversible':True}))
