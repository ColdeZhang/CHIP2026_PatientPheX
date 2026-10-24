"""No-gold full input/resource binding. No model loading, inference or scoring."""
import argparse
import collections
import copy
import json
from pathlib import Path
import sys
from common import checked, deny_gold, module, pairs, read, ref, save

S = Path(__file__).resolve().parent


def save_same(path, value):
    if path.exists():
        assert path.read_bytes() == (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    else: save(path, value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, default=Path('/workspace/patientphex-runs/round2-20261023'))
    ap.add_argument('--grounded', type=Path, default=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation'))
    ap.add_argument('--sap-source', type=Path, default=Path('/workspace/patientphex-worktrees/research-supervised-sapbert-linker/solutions/supervised_sapbert_linker'))
    ap.add_argument('--copy-source', type=Path, default=Path('/workspace/patientphex-worktrees/explore-copy-grounded-decode/solutions/copy_grounded_decode'))
    a = ap.parse_args(); deny_gold(); R = a.runs; G = a.grounded
    out = R / 'research/adapted-quote-linking-v1/F1-prepared-v1'; out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'Preserve prior meaningful prepared artifacts; never overwrite'
    old = read(G / 'quality/plan.json'); norm = read(G / 'normalization/held-finalize-plan.json')
    prep = read(checked(norm['preparation'])); ex = read(checked(prep['extraction_manifest']))
    assert prep['anchors'] == ex['exact_anchors'] == 573 and prep['necessary_public_query_count'] == 254
    assert prep['documents'] == ex['documents'] == 16 and prep['patients'] == ex['patients'] == 39
    assert ex['complete_original39'] and prep['no_gold_fit_or_score']
    assert prep['train32']['sha256'] == '38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    for item in list(prep['sources'].values()) + [prep[k] for k in ['input', 'train32', 'hpo', 'policy']]: checked(item)
    assert prep['policy']['sha256'] == 'b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
    # Reuse all accepted anchors, including those formerly rejected by semantic cosine/margin.
    quotes = [json.loads(x) for x in checked(ex['quotes']).read_text().splitlines()]
    decisions = read(checked(ex['decisions'])); strict_rows = read(checked(prep['strict_decisions']))
    queries = read(checked(prep['queries'])); inputs = [json.loads(x) for x in checked(prep['input']).read_text().splitlines()]
    assert len(quotes) == len(strict_rows) == 573 and len(decisions) == 39
    assert all(r['anchor'] == q for r, q in zip(strict_rows, quotes))
    assert sum(len(d['decision']['anchors']) for d in decisions) == 573
    assert sum(len(d['decision']['rejected']) for d in decisions) == ex['rejected_items']
    assert not any('copy-grounded-decode' in str(x) for x in [norm['preparation'], prep['extraction_manifest'], ex['quotes']])
    sys.path.insert(0, str(G / 'normalization')); import semantic, public_bank
    policy = semantic.load_policy(checked(prep['policy']))
    expected = sorted({r['anchor']['quote'] for r in strict_rows if r['strict']['status'] == 'abstain'
                       and r['strict']['reason'] == policy['semantic_fallback_reason']
                       and semantic.precheck(r['anchor']['quote'], policy) is None})
    assert [x['quote'] for x in queries] == expected and len(expected) == 254
    # Freeze source and actual raw/extraction identities; don't rerun the old parser.
    old_refs = {k: old['actual'][k] for k in ['collection', 'extraction_plan', 'extraction_manifest', 'quotes', 'decisions']}
    for item in old_refs.values(): checked(item)
    collection = read(old_refs['collection']['path'])
    assert collection['all_actual_SHA_verified'] and collection['wrapper']['returncode'] == 0
    extract_plan = read(old_refs['extraction_plan']['path'])
    for item in extract_plan['sources'].values(): checked(item)
    for name in ['complete', 'wrapper', 'responses', 'prompts']: checked(extract_plan['inputs'][name])
    complete = read(extract_plan['inputs']['complete']['path']); assert complete['rows'] == 39
    docs = {str(d['pmc_id']): d for d in inputs}
    strict = module('_adapted_strict_prepare', old['references']['strict'])
    for row in strict_rows:
        q = row['anchor']; assert strict.verify_span(docs[str(q['pmc_id'])], q['quote'], q['offset']) == row['passage_index']
    source_preds = read(a.copy_source / 'evidence/readiness.json')['inputs']
    refs = {k: source_preds[v] for k, v in [('B4', 'B4'), ('S', 'S_with_B'), ('Q', 'Q'), ('C', 'C')]}
    # Actual Q is the pure single prediction, not its later B-union artifact.
    refs['inherited_B'] = source_preds['S_inherited_B']
    sys.path.insert(0, str(Path(old['references']['evaluator']['path']).parent))
    ev = module('_adapted_eval_atoms', old['references']['evaluator'])
    streams = {k: [json.loads(x) for x in checked(v).read_text().splitlines()] for k, v in refs.items()}
    sets = {k: pairs(v, ev) for k, v in streams.items()}
    base = sets['B4']; support = sets['S'] | sets['Q'] | sets['C']
    assert len(base) == 524 and len(streams['B4']) == 16
    assert sum(len(d['entities']) for d in streams['B4']) == 1189 and sum(len(d['association']) for d in streams['B4']) == 39
    assert len(sets['inherited_B']) == 471 and sets['inherited_B'] <= base and sets['inherited_B'] <= sets['S']
    assert support - base == (support - sets['inherited_B']) - base
    assert all({str(d['pmc_id']) for d in rows} == set(docs) for rows in streams.values())
    # New adapted weights are a full train state; metadata does not imply a local model download.
    sap_root = R / 'research/supervised-linker-readiness-20261024'
    actual = sap_root / 'formal-recovery-v1/collected/formal-output/complete.json'; sap = read(actual)
    assert sap['status'] == 'COMPLETE_FRESH3EPOCH219_AND_ALL303_NO_QUALITY' and sap['updates'] == 219
    assert sap['checkpoint']['sha256'] == read(S / 'decision.json')['model']['checkpoint_sha256']
    assert sap['final_bank']['sha256'] == read(S / 'decision.json')['model']['bank_sha256']
    assert sap['fresh_base'] and not sap['failed_formal_state_loaded'] and sap['silent_skipped_updates'] == 0
    labels = sap_root / 'owner-v1/public-rows.json'; label_rows = read(labels)
    hpo = strict.HPO(prep['hpo']['path'])
    assert label_rows == public_bank.public_rows(hpo) and len(label_rows) == 41099
    assert public_bank.digest_json(label_rows) == policy['public_bank_rows_content_sha256']
    assert ref(labels)['sha256'] == '60a27430437a25239768c5d33a6a58d9588dbe5e2f80168c19b9d4c955756b9f'
    sap_plan = read(a.sap_source / 'formal_recovery/formal-plan.json')
    native = read(checked(sap_plan['exact_native_contract']))
    assert native['groups']['bank']['rows'] == 41099 and native['groups']['train']['rows'] == 1160
    source = {p.name: ref(p) for p in [G/'normalization/strict.py', G/'normalization/semantic.py', G/'normalization/public_bank.py']}
    newpolicy = copy.deepcopy(policy)
    newpolicy.update(policy_id='grounded-adapted-sap-conservative-v1',
        score_precision='FP32 dot of L2-normalized FP32 CLS from FP32 parameters under CUDA FP16 autocast; distinct-concept max over all public aliases',
        selection_basis='Root fixed unchanged .90/.03 before new Sap F1 quality; uncalibrated joint control, no threshold scan',
        diagnostic_scope='Complete original G F1 accepted anchors; fixed primary only',
        task_adapted=True, no_task_fit=False, original_policy_sha256=prep['policy']['sha256'],
        checkpoint_sha256=sap['checkpoint']['sha256'], public_bank_sha256=sap['final_bank']['sha256'])
    save_same(S/'policy_v1.json', newpolicy)
    bank_manifest = dict(schema='adapted-Sap-public-alias-bank-v1', task_adapted=True, no_task_fit=False,
        alias_content_uses_train_annotations=False, encoder_training_uses_train32=True,
        hpo_sha256=prep['hpo']['sha256'], rows_content_sha256=public_bank.digest_json(label_rows),
        public_rows=41099, public_labels_sha256=ref(labels)['sha256'], bank=sap['final_bank'], checkpoint=sap['checkpoint'],
        train32=prep['train32'], training_complete=ref(actual), state_ACK=sap['state_ACK'],
        base_model_files=sap_plan['model_files'], pooling='CLS_L2', max_length=32,
        dtype='float32', shape=[41099,768], encoder_parameter_dtype='float32', forward_autocast='CUDA float16',
        actual_remote_vector_SHA_and_norm_gate_required=True, old_public_encoder_certificate_reused=False)
    save_same(S/'bank-manifest.json', bank_manifest)
    from tokenizers import BertWordPieceTokenizer
    vocab = R.parent/'day1-20261023/semantic-neural/sapbert-fp16/vocab.txt'
    assert ref(vocab)['sha256'] == public_bank.MODEL_FILES['vocab.txt']
    tok = BertWordPieceTokenizer(str(vocab), lowercase=True)
    encoded = tok.encode_batch([public_bank.legacy_text(q['quote']) for q in queries])
    ids = [x.ids for x in encoded]; save(out/'query-native-ids.json', ids)
    publish_sources = {**source, 'prepare.py': ref(__file__), 'decision': ref(S/'decision.json')}
    manifest = dict(schema='adapted-quote-CPU-prepared-v1', status='CPU_READY_NO_MODEL_GOLD_OR_QUALITY',
        original_G=old_refs, original_preparation=norm['preparation'], strict_decisions=prep['strict_decisions'],
        queries=prep['queries'], exact_anchors=573, query_count=254, input=prep['input'], train32=prep['train32'], hpo=prep['hpo'],
        sources=publish_sources, original_policy=prep['policy'], new_policy=ref(S/'policy_v1.json'), bank_manifest=ref(S/'bank-manifest.json'),
        public_labels=ref(labels), adapted_training_complete=ref(actual), adapted_training_source=ref(a.sap_source/'formal_recovery/train.py'),
        adapted_training_plan=ref(a.sap_source/'formal_recovery/formal-plan.json'), inherited_native_CPU=ref(checked(sap_plan['exact_native_contract'])),
        query_native_ids=ref(out/'query-native-ids.json'), native_input_scope='Local independent Rust BERT; target actual HF full254 equality required before encoder',
        token_counts=dict(rows=254,tokens=sum(map(len,ids)),max=max(map(len,ids)),over32=sum(len(x)>32 for x in ids)),
        support=dict(refs=refs,counts={k:len(v) for k,v in sets.items()},union=len(support),outside_B4=len(support-base),
                     pure_Q=True,stored_S_inherited_B471_protected=True,old_G_excluded=True),
        byte_helper=old['references']['byte_helper'], original_support=ref(G/'quality/support.py'),
        original_relation_assembler=ref(G/'quality/assemble.py'), grounded_schema=old['references']['schema'], decode=old['references']['decode'],
        evaluator=old['references']['evaluator'],validator=old['references']['validator'],
        no_quality=True,no_model_loaded=True,no_new_fit=True,raw_and_old_parser_not_reexecuted=True)
    save(out/'manifest.json',manifest)
    save(S/'evidence/preparation.json',dict(preparation=ref(out/'manifest.json'),exact_anchors=573,queries=254,
        token_counts=manifest['token_counts'],support=manifest['support'],source=ref(__file__),no_gold_model_or_quality=True))
    print(json.dumps(ref(out/'manifest.json')))


if __name__ == '__main__': main()
