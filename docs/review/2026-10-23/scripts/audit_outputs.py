import json,pathlib,collections,re,hashlib
R=pathlib.Path(__file__).resolve().parents[4]
def read(p): return [json.loads(l) for l in p.open() if l.strip()]
inputs={k:read(R/p) for k,p in [('B','data/B/PatientPheX-V1-B.jsonl'),('train','data/train/PatientPheX-train.jsonl')]}
byid={k:{str(d['pmc_id']):d for d in ds} for k,ds in inputs.items()}
terms={}
for stanza in (R/'data/hp.obo').read_text().split('[Term]')[1:]:
 mid=re.search(r'^id: (HP:\d+)$',stanza,re.M)
 if mid:terms[mid[1]]={'obsolete':'is_obsolete: true' in stanza,'parents':re.findall(r'^is_a: (HP:\d+)',stanza,re.M),'alts':re.findall(r'^alt_id: (HP:\d+)',stanza,re.M),'name':re.search(r'^name: (.*)$',stanza,re.M)[1]}
allids=set(terms);altids={a:p for p,t in terms.items() for a in t['alts']}
branch={'HP:0000118'}
while True:
 add={p for p,t in terms.items() if any(a in branch for a in t['parents'])}-branch
 if not add:break
 branch.update(add)
out={}
for p in sorted((R/'outputs').glob('*.jsonl')):
 ds=read(p);kind='train' if 'train' in p.name else 'B'; expected=byid[kind]
 c=collections.Counter(); examples=collections.defaultdict(list);sec_counts=collections.Counter();invalid_hpos=collections.Counter();mismatch_pats=[]
 def hit(key,value):
  c[key]+=1
  if len(examples[key])<5:examples[key].append(value)
 for d in ds:
  did=str(d.get('pmc_id'));c['docs']+=1
  if did not in expected:hit('unknown_docs',did);continue
  g=expected[did]; ents=d.get('entities',[]); ass=d.get('association',[]);c['entities']+=len(ents)
  if not ents:c['docs_zero_entities']+=1
  expected_pats={a['patient_id'] for a in g['patient']};actual_pats={a['patient_id'] for a in ass}
  c['expected_patients']+=len(expected_pats);c['provided_assoc_records']+=len(ass);c['missing_patients']+=len(expected_pats-actual_pats);c['unexpected_patients']+=len(actual_pats-expected_pats)
  if actual_pats!=expected_pats:mismatch_pats.append({'pmc_id':did,'missing':sorted(expected_pats-actual_pats),'extra':sorted(actual_pats-expected_pats)})
  c['duplicate_patient_records']+=len(ass)-len(actual_pats)
  if d.get('pmid')!=g.get('pmid'):hit('pmid_mismatch',did)
  entitykeys=[]; spanids=collections.defaultdict(set); ent_hpos=set()
  for e in ents:
   h=str(e['identifier']);s=e['offset'];l=e['length'];t=e['text'];end=s+l
   entitykeys.append((s,l,h));spanids[(s,l)].add(h)
   if e.get('note')=='NO':c['negative_note']+=1
   if h=='-1':c['unmapped_entities']+=1
   if ';' in h:c['multi_id_entities']+=1
   hpos=h.split(';');c['entity_hpo_units']+=len(hpos);ent_hpos.update(hpos)
   for hp in hpos:
    if hp=='-1':continue
    if hp not in allids and hp not in altids:invalid_hpos[hp]+=1;c['unknown_hpo_units']+=1
    primary=altids.get(hp,hp)
    if hp in altids:c['alt_hpo_units']+=1
    if primary in terms and terms[primary]['obsolete']:c['obsolete_hpo_units']+=1
    if primary in terms and primary not in branch:c['outside_branch_units']+=1
   if len(t)!=l:hit('text_length_mismatch',{'pmc':did,'entity':e})
   covering=[sec for sec in g['full_text'] if sec['offset']<=s and end<=sec['offset']+len(sec['text'])]
   if not covering:hit('span_outside_passages',{'pmc':did,'entity':e});continue
   sec=covering[0];sec_counts[sec['section_type']]+=1
   raw=sec['text'][s-sec['offset']:end-sec['offset']]
   if raw!=t:hit('text_span_mismatch',{'pmc':did,'entity':e,'source':raw})
   if sec['section_type'] in ['REF','REFS','COMP_INT','AUTH_CONT','ACK','ACK_FUND','SUPPL','KEYWORD']:c['entities_in_nonbody_or_metadata']+=1
  c['duplicate_entities']+=len(entitykeys)-len(set(entitykeys))
  c['same_span_multiple_identifier']+=sum(len(hps)>1 for hps in spanids.values())
  for a in ass:
   ph=a['phenotype'];c['links']+=len(ph)
   if not ph:c['empty_patients']+=1
   c['duplicate_assoc_phenotypes']+=len(ph)-len(set(ph))
   for hp in ph:
    if hp not in ent_hpos:c['association_without_entity_id']+=1
    if hp.startswith('HP:') and hp not in allids and hp not in altids:c['unknown_assoc_hpo']+=1
    if ';' in hp:c['unsplit_compound_assoc']+=1
 docids=[str(d['pmc_id']) for d in ds]
 c['duplicate_docs']=len(docids)-len(set(docids));c['missing_docs']=len(set(expected)-set(docids));c['order_equal_input']=int(docids==[str(d['pmc_id']) for d in inputs[kind]])
 out[p.name]={'dataset':kind,'counts':dict(c),'sections':dict(sec_counts),'unknown_hpos':dict(invalid_hpos),'patient_mismatch_examples':mismatch_pats[:8],'examples':dict(examples),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
pathlib.Path('/tmp/patientphex-output-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for name,a in out.items(): print(name,json.dumps(a['counts'],ensure_ascii=False))
for name,a in out.items():
 if a['examples'] or a['patient_mismatch_examples'] or a['unknown_hpos']:print('\nEXAMPLES',name,json.dumps({k:a[k] for k in ['examples','patient_mismatch_examples','unknown_hpos']},ensure_ascii=False))
