"""Freeze allowed input-only docs in F2,F3,F1 file order; never read gold."""
import pathlib,json,hashlib,sys,datetime
P=pathlib.Path(__file__).parent;R=P.parent;sys.path.insert(0,str(R))
from anchor_mapping import original_anchors,canonical_sha,text_view
old=json.loads((R/'anchor-census.json').read_text());allowed=set(old['groups']['train32']['document_ids']);rows=[];sources=[];seen=set();counts={}
for fold in [2,3,1]:
 source=next(x for x in old['sources'] if '/fold-'+str(fold)+'/' in x['path']);p=pathlib.Path(source['path']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==source['sha256'];sources.append(source)
 for ln,line in enumerate(b.decode().splitlines(),1):
  doc=json.loads(line);pid=str(doc['pmc_id']);assert pid not in seen;seen.add(pid);original_anchors(doc);assert (pid in allowed)==(fold in [2,3]);text,segs=text_view(doc)
  rows.append({'group':'train32' if fold in [2,3] else 'F1','fold':fold,'source_line':ln,'source_file_sha256':source['sha256'],'input_canonical_sha256':canonical_sha(doc),'document':doc})
  counts[pid]={'group':rows[-1]['group'],'chars':len(text),'passages':len(segs),'patients':len(doc['patient']),'original_mentions':sum(len(p['mention']) for p in doc['patient'])}
assert len(rows)==48 and set(str(x['document']['pmc_id']) for x in rows if x['group']=='train32')==allowed
packed=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in rows).encode();(P/'input-only48.jsonl').write_bytes(packed)
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'strict train32 F2+F3, disjoint F1; input-only, no gold entities/relations','sources':sources,'packing_order':'F2 lines, F3 lines, F1 lines','input_file':{'name':'input-only48.jsonl','sha256':hashlib.sha256(packed).hexdigest(),'bytes':len(packed)},'document_ids_in_order':[str(x['document']['pmc_id']) for x in rows],'documents':counts,'first_probe_pmc_id':str(rows[0]['document']['pmc_id']),'groups':{'train32':32,'F1':16}}
(P/'input-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'bytes':len(packed),'sha256':manifest['input_file']['sha256'],'first_probe':manifest['first_probe_pmc_id'],'chars_first':next(iter(counts.values()))['chars'],'chars_total':sum(v['chars'] for v in counts.values()),'chars_max':max(v['chars'] for v in counts.values())}))
