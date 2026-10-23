#!/usr/bin/env python3
"""JATS geometry and conservative patient identity feasibility. No gold or predictions.

Copied from round2_validation analyze_tables.py (2026-10-23), original SHA256
0966b926b41942576bd329065de05a5755749357e2be52fc0e14f3c8be3a61a5.
Only CLI removed; parser/identity functions unchanged.
"""
import argparse,collections,datetime,hashlib,json,pathlib,re,unicodedata,xml.etree.ElementTree as ET

def tag(e):return e.tag.rsplit('}',1)[-1]
def plain(e):return ' '.join(''.join(e.itertext()).split()) if e is not None else ''
def identity_text(e):
 def visit(x):
  s=x.text or ''
  for child in x:
   if tag(child) in {'break','br'}:s+=' '
   if not (tag(child)=='xref' and child.get('ref-type') in {'fn','table-fn','bibr'}):s+=visit(child)
   s+=child.tail or ''
  return s
 return ' '.join(visit(e).split())
def norm(s):
 s=unicodedata.normalize('NFKC',s).casefold()
 s=''.join('-' if unicodedata.category(c)=='Pd' or c in '\u2212\u00ad' else c for c in s)
 return ' '.join(s.split())
def identifier(s):
 s=norm(s);s=re.sub(r'^(?:patient|case|subject|individual)\s+','',s)
 # Keep punctuation distinguishing pedigrees; never strip an internal or numerical token.
 s=re.sub(r'\s*([:.\-])\s*',r'\1',s)
 if re.fullmatch(r'[ivxlcdm]+[:.\-]\d+(?:[:.\-]\d+)*',s):return 'pedigree:'+s
 if re.fullmatch(r'\d+[a-z]+\d*',s):return 'alphanumeric:'+s
 if re.fullmatch(r'\d+',s):return 'number:'+s
 return None
GENERIC={'patient','subject','case','individual','she','he','her','his','girl','boy','woman','man','infant','proband','this patient','the patient','the infant','a female','a male'}
SUBJECT_COLUMNS={'subject','patient','patient number','case','individual','patient id','subject id','case number','case id'}
HISTORICAL=re.compile(r'previous(?:ly)?|reported|literature|et al|historical',re.I)
CURRENT=re.compile(r'\b(?:this case|present case|current study|present study|this study|our patient)\b',re.I)

def parse_table(table,wrap,index):
 parents={c:p for p in table.iter() for c in p};trs=[]
 for tr in table.iter():
  if tag(tr)!='tr':continue
  parent=parents[tr]
  while parent is not table and tag(parent)!='table':parent=parents[parent]
  if parent is table:trs.append(tr)
 groups=[]
 for tr in trs:
  parent=parents[tr]
  while parent is not table and tag(parent) not in {'thead','tbody','tfoot'}:parent=parents[parent]
  groups.append((tag(parent) if parent is not table else 'direct',id(parent)))
 grid={};cells=[];issues=[]
 for r,tr in enumerate(trs):
  c=0
  for element in tr:
   if tag(element) not in {'th','td'}:continue
   while (r,c) in grid:c+=1
   rs=int(element.get('rowspan','1'));cs=int(element.get('colspan','1'))
   if rs==0:rs=sum(g==groups[r] for g in groups[r:])
   if rs<1 or cs<1:raise ValueError('Nonpositive span')
   if r+rs>len(trs):raise ValueError('Span exceeds rows')
   cid=f'r{r}c{c}'
   cell={'id':cid,'row':r,'column':c,'rowspan':rs,'colspan':cs,'row_group':groups[r][0],'tag':tag(element),'attributes':dict(element.attrib),'text':plain(element),'identity_text':identity_text(element),'xml':ET.tostring(element,encoding='unicode'),'xrefs':[dict(e.attrib,text=plain(e)) for e in element.iter() if tag(e)=='xref']}
   cells.append(cell)
   for rr in range(r,r+rs):
    for cc in range(c,c+cs):
     if (rr,cc) in grid:raise ValueError('Overlapping cells')
     grid[(rr,cc)]=cid
   c+=cs
 cols=max((c+1 for r,c in grid),default=0);matrix=[[grid.get((r,c)) for c in range(cols)] for r in range(len(trs))]
 for r,row in enumerate(matrix):
  if any(c is None for c in row):issues.append({'kind':'ragged_row','row':r,'missing_columns':[i for i,c in enumerate(row) if c is None]})
 byid={c['id']:c for c in cells};headers=[]
 for c in range(cols):
  path=[]
  for r in range(len(trs)):
   cid=grid.get((r,c));cell=byid.get(cid,{})
   if cid and (groups[r][0]=='thead' or (cell.get('tag')=='th' and cell['attributes'].get('scope') in {'col','colgroup'})) and cid not in path:path.append(cid)
  headers.append({'column':c,'cell_ids':path,'texts':[byid[cid]['identity_text'] for cid in path]})
 foot=[]
 for el in wrap:
  if tag(el)=='table-wrap-foot':foot.append({'text':plain(el),'xml':ET.tostring(el,encoding='unicode')})
 return {'table_wrap_id':wrap.get('id'),'table_index':index,'label':plain(wrap.find('label')),'caption':plain(wrap.find('caption')),'rows':len(trs),'columns':cols,'cells':cells,'grid_cell_ids':matrix,'column_header_paths':headers,'footnotes':foot,'geometry_issues':issues,'rowspan_cells':sum(c['rowspan']>1 for c in cells),'colspan_cells':sum(c['colspan']>1 for c in cells)}

def candidates(table):
 cells={c['id']:c for c in table['cells']};result=[]
 for h in table['column_header_paths']:
  if not h['cell_ids']:continue
  cell=cells[h['cell_ids'][-1]]
  if cell['colspan']==1 and cell['identity_text']:
   result.append({'axis':'column','index':h['column'],'cell_id':cell['id'],'text':cell['identity_text'],'header_path':h['texts'],'citation_in_header':any(x.get('ref-type')=='bibr' for cid in h['cell_ids'] for x in cells[cid]['xrefs'])})
  if norm(cell['identity_text']) in SUBJECT_COLUMNS:
   for row in table['grid_cell_ids']:
    cid=row[h['column']]
    if not cid:continue
    x=cells[cid]
    if x['row_group']=='thead' or x['rowspan']!=1 or x['colspan']!=1 or not x['identity_text']:continue
    result.append({'axis':'subject_row','index':x['row'],'cell_id':cid,'text':x['identity_text'],'header_path':h['texts'],'citation_in_header':False})
 return result

def match(table,patients,alias_source="mentions"):
 candidates_=candidates(table);results=[];rejected=[]
 for cand in candidates_:
  value=norm(cand['text']);key=identifier(cand['text']);matches=[]
  for patient in patients:
   aliases={m['text'] for m in patient['mention']} if alias_source!='official_o_id_only' else set()
   if alias_source!='mentions' and patient['patient_id'].startswith('O'):aliases.add(patient['patient_id'][1:])
   exact=[s for s in aliases if norm(s)==value and value not in GENERIC and not value.isdigit()]
   ids=[s for s in aliases if identifier(s)==key] if key else []
   if exact or ids:matches.append({'patient_id':patient['patient_id'],'level':'lexical_exact' if exact else 'anchored_identifier','matched_aliases':sorted(exact or ids),'alias_source':alias_source,'key':key})
  if not matches:continue
  reasons=[]
  if len(matches)!=1:reasons.append('multiple_input_patients_share_label')
  if cand['citation_in_header']:reasons.append('bibliographic_header')
  if HISTORICAL.search(' '.join(cand['header_path'])+' '+table['caption']):reasons.append('historical_or_reported_context')
  record=dict(cand,matches=matches)
  if reasons:rejected.append(dict(record,reasons=reasons))
  else:results.append(record)
 # A patient must resolve to one column or subject row within each table, not merely a unique alias.
 counts=collections.Counter(m['matches'][0]['patient_id'] for m in results)
 accepted=[]
 for m in results:
  if counts[m['matches'][0]['patient_id']]>1:rejected.append(dict(m,reasons=['multiple_locations_for_patient_in_same_table']))
  else:accepted.append(m)
 return {'accepted':accepted,'rejected':rejected,'unmapped_generic_patient_columns':[c for c in candidates_ if re.search(r'\bpatient\b',norm(c['text'])) and not identifier(c['text'])],'current_case_markers':[{'cell_id':c['id'],'text':c['identity_text']} for c in table['cells'] if CURRENT.search(c['identity_text'])],'historical_context':bool(HISTORICAL.search(table['caption']))}

