"""One frozen F2+F3 train32 / F1 input-only anchor census; never access gold."""
import argparse,collections,hashlib,json,pathlib,re
from anchor_mapping import original_anchors

def ref(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 a=argparse.ArgumentParser();a.add_argument('--data-root',required=True);a.add_argument('--training-manifest',required=True);a.add_argument('--output',required=True);args=a.parse_args();root=pathlib.Path(args.data_root);manifest=pathlib.Path(args.training_manifest);allowed=set(json.loads(manifest.read_text())['training_document_ids']);groups={};sources=[];details=[]
 for group,folds in [('train32',[2,3]),('F1',[1])]:
  counts=collections.Counter();ids=[]
  for fold in folds:
   p=root/f'fold-{fold}'/'validation16.input.jsonl';sources.append(ref(p))
   for line in p.read_text().splitlines():
    doc=json.loads(line);assert not doc.get('entities') and not doc.get('association');pid=str(doc['pmc_id']);assert pid not in ids;ids.append(pid);anchors=original_anchors(doc);literal=collections.defaultdict(set);normalized=collections.defaultdict(set);counts.update(documents=1,patients=len(doc['patient']),unique_anchor_spans=len(anchors),multi_patient_exact_spans=sum(len(v)>1 for v in anchors.values()))
    for person in doc['patient']:
     mentions=person.get('mention',[]);counts['patients_without_mentions']+=not mentions;counts['original_mention_records']+=len(mentions)
     for m in mentions:
      literal[m['text']].add(str(person['patient_id']));normalized[re.sub(r'\s+',' ',m['text']).casefold()].add(str(person['patient_id']))
    counts['literal_alias_types']+=len(literal);counts['literal_multi_patient_alias_types']+=sum(len(v)>1 for v in literal.values());counts['normalized_multi_patient_alias_types_diagnostic_only']+=sum(len(v)>1 for v in normalized.values())
    counts['docs_with_literal_alias_conflicts']+=any(len(v)>1 for v in literal.values());counts['docs_with_exact_span_conflicts']+=any(len(v)>1 for v in anchors.values())
    details.append({'group':group,'pmc_id':pid,'original_mention_records':sum(len(p.get('mention',[])) for p in doc['patient']),'exact_span_conflicts':[{'offset':s,'length':e-s,'patient_ids':sorted(v)} for (s,e),v in anchors.items() if len(v)>1],'literal_alias_conflicts':[{'text':k,'patient_ids':sorted(v)} for k,v in literal.items() if len(v)>1]})
  if group=='train32':assert set(ids)==allowed
  else:assert not set(ids)&allowed
  groups[group]={'document_ids':sorted(ids),'counts':dict(counts)}
 out={'status':'input_only_complete_no_gold_no_NN','groups':groups,'sources':sources,'training_document_id_manifest':ref(manifest),'documents':details,'safe_mapping':'Only predicted clusters with exactly one original patient ID from exact original mention intervals; multiple IDs/zero IDs/invalid mapping rejected. Same alias at disjoint offsets does not itself imply conflict.','not_quality_or_cluster_coverage':True}
 pathlib.Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v['counts'] for k,v in groups.items()}))
if __name__=='__main__':main()
