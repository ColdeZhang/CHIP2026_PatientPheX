"""Frozen input-only A exact/B containment anchoring; no labels or learned thresholds."""
from anchor_mapping import original_anchors,map_clusters,inject_mentions

def map_clusters_ab(doc,text,segments,clusters,source,mode):
 if mode not in ('A','B'):raise ValueError('mode must be A or B')
 mapped=map_clusters(doc,text,segments,clusters,source)
 mapped['mapping_mode']=mode
 if mode=='A':return mapped
 anchors=original_anchors(doc)
 for c in mapped['clusters']:
  # Union over every given anchor contained by every model mention in this cluster.
  evidence=[];ids=set();shared=False
  for m in c['mentions']:
   for (s,e),pids in anchors.items():
    if m['offset']<=s<e<=m['offset']+m['length']:
     ids.update(pids);shared|=len(pids)>1
     evidence.append({'model_mention':m,'original_anchor_offset':s,'original_anchor_length':e-s,'patient_ids':sorted(pids)})
  c['containment_evidence']=evidence;c['anchor_patient_ids']=sorted(ids)
  c['status']='invalid_span' if c['invalid'] else ('conflicting_patients' if shared or len(ids)>1 else ('unique_patient' if len(ids)==1 else 'unanchored'))
  c['patient_id']=next(iter(ids)) if c['status']=='unique_patient' else None
 mapped['claim']='input-only candidate identity, not clinical attribution; known anchors only; unknown people remain a risk'
 return mapped

def inject_ab(doc,mapped):
 result,sidecar=inject_mentions(doc,mapped)
 sidecar['mapping_mode']=mapped['mapping_mode']
 sidecar['policy']='A exact/B full containment of given anchors; unique patient over whole cluster; reject shared coordinates, invalid/passages, all overlap with original anchors and overlapping proposals; no string alias broadcast; no relation assertion'
 return result,sidecar
