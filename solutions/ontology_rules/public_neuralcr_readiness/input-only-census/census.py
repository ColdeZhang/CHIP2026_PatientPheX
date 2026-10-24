"""Only the unchanged pre-NN NCR enumeration on frozen four-field train64 inputs."""
from pathlib import Path
import ast, collections, copy, datetime, hashlib, json, math, re, time
R=Path(__file__).resolve().parent
plan=json.loads((R/'plan.json').read_text());started=time.monotonic()
sha=lambda b:hashlib.sha256(b).hexdigest()
def load_ref(ref):
    b=Path(ref['path']).read_bytes();assert len(b)==ref['bytes'] and sha(b)==ref['sha256'];return b
source=load_ref(plan['native_source']).decode();tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NCR')
native=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='annotate_text')
stop=next(i for i,n in enumerate(native.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='matches' for t in n.targets))
body=copy.deepcopy(native.body[:stop]);candidate=copy.deepcopy(native);candidate.name='enumerate_native';candidate.body=body+[ast.Return(ast.Tuple([ast.Name('candidates',ast.Load()),ast.Name('candidates_info',ast.Load())],ast.Load()))]
plain=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('is_number','tokenize')]
module=ast.fix_missing_locations(ast.Module(body=plain+[candidate],type_ignores=[]));scope={'re':re};exec(compile(module,'<exact NCR pre-NN AST>','exec'),scope)
assert [ast.dump(x,include_attributes=False) for x in body]==[ast.dump(x,include_attributes=False) for x in native.body[:stop]]
enumerate_native=scope['enumerate_native'];tokenize=scope['tokenize'];norm=re.compile('[\\\\/\r\n\t-]')
contracts=[]
for text,expected in [('Seizures.',[(0,8)]),('A B C',[(0,1),(0,3),(0,5),(2,3),(2,5),(4,5)]),('ab-cd',[(0,2),(0,5),(3,5)]),('aa  bb',[(0,2),(4,6)]),('Café x',[(0,3),(5,6)]),('',[])]:
    queries,spans=enumerate_native(None,text)
    assert [(a,b) for a,b,c in spans]==expected
    assert all(norm.sub(' ',text[a:b])==q for q,(a,b,c) in zip(queries,spans))
    contracts.append({'text':text,'expected_spans':expected,'actual_candidates':len(queries),'offset_roundtrip':True})

input_bytes=load_ref(plan['input']);docs=[json.loads(x) for x in input_bytes.splitlines()]
assert len(docs)==64 and len({x['pmc_id'] for x in docs})==64
assert all(set(x)=={'pmc_id','pmid','patient','full_text'} for x in docs)
ids={x['pmc_id'] for x in docs};old_keys=set();old_rows=0
for ref in plan['old_public_canonical']:
    for line in load_ref(ref).splitlines():
        o=json.loads(line);assert o['pmc_id'] in ids
        old_keys.add((o['pmc_id'],o['passage_index'],o['local_start'],o['local_end']));old_rows+=1
unique=set();raw_unique=set();ngram=collections.Counter();per_doc=[];overlap=set();all_candidates=0;all_passages=0;zero_passages=0;characters=0
for doc in docs:
    count=0;doc_unique=set();empty=0
    for pi,p in enumerate(doc['full_text']):
        assert set(p)<= {'section_type','type','offset','text'} and isinstance(p['offset'],int)
        text=p['text'];characters+=len(text);all_passages+=1
        queries,spans=enumerate_native(None,text);assert len(queries)==len(spans)
        assert len(set((a,b) for a,b,c in spans))==len(spans)
        if not queries:zero_passages+=1;empty+=1
        for q,(a,b,c) in zip(queries,spans):
            assert 0<=a<b<=len(text) and norm.sub(' ',text[a:b])==q
            tokens=tokenize(q)[:49];assert 1<=len(tokens)<=7
            key=' '.join(tokens);assert tokenize(key)==tokens
            unique.add(key);raw_unique.add(q);doc_unique.add(key);ngram[len(tokens)]+=1
            span=(doc['pmc_id'],pi,a,b)
            if span in old_keys:overlap.add(span)
        count+=len(queries)
    all_candidates+=count
    per_doc.append({'pmc_id':doc['pmc_id'],'passages':len(doc['full_text']),'raw_candidates':count,'unique_normalized_queries':len(doc_unique),'zero_candidate_passages':empty})
query_path=R/'unique-normalized-queries.jsonl'
with query_path.open('x') as f:
    for q in sorted(unique):f.write(json.dumps({'query':q},ensure_ascii=False,separators=(',',':'))+'\n')
report={'status':'pass_input_only_native_enumeration_no_model_or_quality',
        'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_sha256':sha(Path(__file__).read_bytes()),'plan_sha256':sha((R/'plan.json').read_bytes()),
        'exact_native_pre_get_match_AST':True,'extracted_body_AST_sha256':sha(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()),
        'synthetic_contracts':contracts,'input_documents':64,'input_fields':['pmc_id','pmid','patient','full_text'],
        'input_sha256':plan['input']['sha256'],'passages':all_passages,'characters':characters,
        'raw_candidate_occurrences':all_candidates,'unique_raw_case_sensitive_strings':len(raw_unique),
        'unique_phrase2vec_token_keys':len(unique),'ngram_length_occurrences':dict(sorted(ngram.items())),
        'zero_candidate_passages':zero_passages,'all_candidate_offsets_roundtrip':True,
        'old_public_canonical_coordinate_comparison':{'source_rows':old_rows,'unique_coordinates':len(old_keys),
            'coordinates_represented_in_NCR_raw_enumeration':len(overlap),
            'old_coordinates_absent_from_NCR_raw_enumeration':len(old_keys-overlap),
            'NCR_raw_coordinates_outside_old_public_canonical':all_candidates-len(overlap),
            'scope':'Raw coordinate supply only, not accepted NCR HPOs, true positives, or recall against labels.'},
        'cache_reuse':{'other_model_neural_logits_reusable':0,
            'reason':'Different fastText/CNN/HPO model; identical text does not make PhenoTagger outputs reusable.',
            'old_raw_key_intersection_measured':False,
            'old_raw_key_limitation':'Complete existing raw 244938-key cache is remote-only; no extra MCP retrieval during outage. Canonical coordinate comparison above is independently available locally.',
            'NCR_internal_dedup_key':'Exact tokenize(phrase)[:49] tuple, represented by joined ASCII tokens; all generated phrases <=7 tokens. No NN executed.'},
        'workload_only':{'unique_batches_at_native512':math.ceil(len(unique)/512),'max_sequence_length':50,
            'word_vector_dim':100,'ensembles':10,'ontology_output_columns':14261,
            'native_four_synthetic_pending':True,'seconds_projection':None,
            'reason':'No matching actual native batch throughput measured; enum CPU time is not NN inference time.'},
        'per_document':per_doc,'artifact':{'path':str(query_path),'bytes':query_path.stat().st_size,'sha256':sha(query_path.read_bytes())},
        'competition_labels_read':False,'task_NN_run':False,'new_HPO_predictions':False,'quality_evaluation':False,
        'elapsed_seconds':time.monotonic()-started}
(R/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['status','raw_candidate_occurrences','unique_raw_case_sensitive_strings','unique_phrase2vec_token_keys','zero_candidate_passages','old_public_canonical_coordinate_comparison','elapsed_seconds']}))
