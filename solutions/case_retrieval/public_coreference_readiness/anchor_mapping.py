"""Input-only exact-span patient anchoring. No model, gold, alias expansion or relation decisions."""
import copy,hashlib,json

def canonical_sha(obj):
 return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def original_anchors(doc):
 if doc.get('entities') or doc.get('association'):raise ValueError('input-only document required')
 passages=sorted(doc['full_text'],key=lambda p:int(p['offset']))
 for a,b in zip(passages,passages[1:]):
  if int(a['offset'])+len(a['text'])>int(b['offset']):raise ValueError('overlapping passages')
 anchors={};patients=[]
 for patient in doc.get('patient',[]):
  pid=str(patient['patient_id'])
  if pid in patients:raise ValueError('duplicate patient ID')
  patients.append(pid)
  for m in patient.get('mention',[]):
   start=int(m['offset']);end=start+int(m.get('length',len(m['text'])))
   matches=[p for p in passages if int(p['offset'])<=start<end<=int(p['offset'])+len(p['text'])]
   if len(matches)!=1 or matches[0]['text'][start-int(matches[0]['offset']):end-int(matches[0]['offset'])]!=m['text']:raise ValueError('invalid original anchor coordinates')
   anchors.setdefault((start,end),set()).add(pid)
 return anchors

def text_view(doc):
 """Explicit segment map for a virtual full-document string; no fabricated global indices."""
 original_anchors(doc);pieces=[];segments=[];cursor=0
 for p in sorted(doc['full_text'],key=lambda p:int(p['offset'])):
  if pieces:pieces.append('\n\n');cursor+=2
  pieces.append(p['text']);segments.append({'local_start':cursor,'local_end':cursor+len(p['text']),'global_start':int(p['offset'])});cursor+=len(p['text'])
 return ''.join(pieces),segments

def project_span(text,segments,start,end):
 if not isinstance(start,int) or not isinstance(end,int) or not 0<=start<end<=len(text):raise ValueError('invalid model span')
 hits=[s for s in segments if s['local_start']<=start<end<=s['local_end']]
 if len(hits)!=1:raise ValueError('span crosses passage/separator or has no mapping')
 seg=hits[0];g=seg['global_start']+start-seg['local_start']
 return {'offset':g,'length':end-start,'text':text[start:end]}

def map_clusters(doc,text,segments,clusters,source):
 anchors=original_anchors(doc);expected_text,expected_segments=text_view(doc)
 if text!=expected_text or segments!=expected_segments:raise ValueError('nonreversible document view')
 out=[]
 for idx,cluster in enumerate(clusters):
  spans=[];bad=[]
  for interval in cluster:
   try:spans.append(project_span(text,segments,*interval))
   except (ValueError,TypeError) as e:bad.append(str(e))
  ids=set().union(*(anchors.get((m['offset'],m['offset']+m['length']),set()) for m in spans)) if spans else set()
  status='invalid_span' if bad else ('unique_patient' if len(ids)==1 else ('conflicting_patients' if len(ids)>1 else 'unanchored'))
  out.append({'cluster_id':idx,'status':status,'patient_id':next(iter(ids)) if status=='unique_patient' else None,'anchor_patient_ids':sorted(ids),'mentions':spans,'invalid':bad,'source':source})
 return {'pmc_id':str(doc['pmc_id']),'original_input_sha256':canonical_sha(doc),'view_sha256':hashlib.sha256(text.encode()).hexdigest(),'segments':segments,'clusters':out,'claim':'identity candidates only; no clinical attribution assertion'}

def inject_mentions(doc,mapped):
 """Append uniquely anchored offsets; retain original mention order. Emit reversible sidecar."""
 if mapped['original_input_sha256']!=canonical_sha(doc) or mapped['pmc_id']!=str(doc['pmc_id']):raise ValueError('source mismatch')
 original_ranges=list(original_anchors(doc));result=copy.deepcopy(doc);lookup={str(p['patient_id']):p for p in result['patient']};added=[]
 # An interval proposed for multiple patients is rejected from every patient.
 proposals={}
 for c in mapped['clusters']:
  if c['status']!='unique_patient':continue
  for m in c['mentions']:proposals.setdefault((m['offset'],m['length']),{}).setdefault(c['patient_id'],[]).append(c['cluster_id'])
 for c in mapped['clusters']:
  if c['status']!='unique_patient':continue
  patient=lookup[c['patient_id']];known={(int(m['offset']),int(m.get('length',len(m['text'])))) for m in patient.get('mention',[])}
  for m in c['mentions']:
   key=(m['offset'],m['length'])
   if key in known or len(proposals[key])!=1:continue
   # Original evidence always wins; no added span may hide an original marker.
   if any(m['offset']<end and start<m['offset']+m['length'] for start,end in original_ranges):continue
   # Reject every overlapping proposal, rather than silently picking a winner.
   if any(k!=key and m['offset']<k[0]+k[1] and k[0]<m['offset']+m['length'] for k in proposals):continue
   patient.setdefault('mention',[]).append(copy.deepcopy(m));known.add(key);added.append({'patient_id':c['patient_id'],'mention':m,'cluster_id':c['cluster_id'],'source':c['source']})
 original_anchors(result)
 return result,{'original_input_sha256':canonical_sha(doc),'derived_input_sha256':canonical_sha(result),'added_mentions':added,'policy':'exact original-anchor match; unique patient; no string alias broadcast; original order preserved; no relation assertion'}
