"""CPU-only full64 response-code readiness; originaldev labels never opened."""
from pathlib import Path
import gzip, hashlib, importlib.util, json, sys, tarfile, time

S = Path(__file__).resolve().parent
ROOT = S.parent
R = Path('/workspace/patientphex-runs/round2-20261023')
B = R / 'research/patient-response-code-full64-readiness-v1'
Q = R / 'research/patient-response-code-readiness-20261024'
OLD = R / 'case_retrieval/qwen35-preparation/full64-relation-cpu-preparation'
H = R / 'research/nonthinking-consensus-full64-readiness-20261024/preparation-v1'
sys.path.insert(0, str(S / 'cpu'))
from response_codec import transform, decode_response, TABLE_PREFIX

sha = lambda b: hashlib.sha256(b).hexdigest()
def ref(p):
    p = Path(p); b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=sha(b))
def save(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')
def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]
def checked(p, digest):
    r = ref(p); assert r['sha256'] == digest, str(p)
    return r

def main():
    started = time.monotonic(); B.mkdir(exist_ok=False)
    prep = json.loads((OLD / 'preparation-manifest.json').read_text())
    hp = json.loads((H / 'complete.json').read_text())
    held_input = Path(hp['refs']['pure_input']['path'])
    documents = rows(held_input)
    assert all(not {'entities', 'association'} & x.keys() for x in documents)
    train_ids = set(prep['training_document_ids']); held_ids = {str(x['pmc_id']) for x in documents}
    assert len(train_ids) == 64 and len(held_ids) == 16 and not train_ids & held_ids
    assert set(prep['excluded_document_ids']) == held_ids
    assert prep['all_target_documents_excluded_from_own_generators']
    assert prep['gold_added_candidates'] == 0 and not prep['gold_in_prompts']
    assert hp['all_task_fitted_ancestors_exclude_originaldev']
    assert set(hp['generation_train_ids']) == train_ids and set(hp['excluded_originaldev_ids']) == held_ids
    ancestry = [ref(OLD / 'preparation-manifest.json'), ref(H / 'complete.json')]
    for r in hp['refs'].values():
        actual = checked(r['path'], r['sha256']); assert actual['bytes'] == r['bytes']; ancestry.append(actual)
    # Training labels are the already frozen train64 targets, never held labels.
    ancestry.append(checked('/workspace/patientphex-data/day1/train64.jsonl', prep['train64_sha256']))
    for fold in prep['provenance']:
        fit_ids = set(fold['train_ids']); own_ids = set(fold['input_ids'])
        assert len(fit_ids) == 48 and len(own_ids) == 16 and not fit_ids & own_ids
        assert fit_ids | own_ids == train_ids and not (fit_ids | own_ids) & held_ids
        foldbase = Path('/workspace/patientphex-data/round2-inner4') / ('fold-' + str(fold['fold']))
        ancestry += [checked(foldbase / 'train48.jsonl', fold['train_sha256']), checked(foldbase / 'validation16.input.jsonl', fold['input_sha256'])]
        ancestry.append(checked(fold['prediction_path'], fold['prediction_sha256']))
    train_source = OLD / 'train.jsonl'; held_source = H / 'expanded64.jsonl'
    checked(train_source, prep['training_sha256']); checked(held_source, hp['expanded']['sha256'])
    original = {'train': rows(train_source), 'held': rows(held_source)}
    assert len(original['train']) == 265 and len(original['held']) == 64
    assert {str(x['pmc_id']) for x in original['train']} == train_ids
    assert {str(x['pmc_id']) for x in original['held']} == held_ids
    assert all('answer' in r for r in original['train']) and all('answer' not in r for r in original['held'])
    # Reuse the exact prior full-report append constructor only to verify existing
    # pure inputs. Never invoke the old character-budget renderer.
    constructor = Path(hp['refs']['constructor']['path'])
    sys.path.insert(0, str(constructor.parent))
    spec = importlib.util.spec_from_file_location('full64_append_contract', constructor)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    short_held = rows(hp['refs']['original_short_prompts']['path'])
    reconstructed, _, _ = module.append_reports(short_held, documents, False)
    assert reconstructed == original['held']
    groups = {}; audit = []
    for group, data in original.items():
        output = []; maps = {}
        for i, row in enumerate(data):
            new = transform(row); output.append(new)
            user = row['messages'][1]['content']
            assert new['messages'][1]['content'][:len(user)].encode() == user.encode()
            assert new['messages'][1]['content'][len(user):].startswith(TABLE_PREFIX)
            assert {k:v for k,v in new.items() if k not in ['messages','answer','response_code_map','response_patient_codes','response_representation']} == {k:v for k,v in row.items() if k not in ['messages','answer']}
            if group == 'train':
                assert json.dumps(decode_response(new['answer'], new), ensure_ascii=False, separators=(',', ':')) == row['answer']
            doc = str(row['pmc_id'])
            assert doc not in maps or maps[doc] == new['response_code_map']
            maps[doc] = new['response_code_map']
            audit.append(dict(group=group,index=i,row_id=row['row_id'],pmc_id=doc,original_user_utf8_sha256=sha(user.encode()),preserved_prefix_sha256=sha(new['messages'][1]['content'][:len(user)].encode()),all_original_fields_except_explicit_system_table_target_exact=True,target_inverse_exact=group=='train'))
        source = train_source if group == 'train' else held_source
        (B / (group + '.response-code.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output))
        (B / (group + '.original.jsonl.gz')).write_bytes(gzip.compress(source.read_bytes(),mtime=0))
        groups[group] = dict(rows=len(data),documents=len(maps),patients=sum(len(m) for m in maps.values()),CID_values=sum(len(r['code_map']) for r in data),patient_concept_pairs=sum(len(r['code_map'])*len(r['patient_ids']) for r in data),source=ref(source),output=ref(B/(group+'.response-code.jsonl')),compressed_original=ref(B/(group+'.original.jsonl.gz')))
    assert (groups['train']['patients'],groups['train']['CID_values'],groups['train']['patient_concept_pairs']) == (173,1908,5328)
    assert (groups['held']['patients'],groups['held']['CID_values'],groups['held']['patient_concept_pairs']) == (36,466,1006)
    save(B/'row-contract.json', audit)
    for name in ['native_cpu.py','response_codec.py','qwen35_smoke.py']:
        (B/name).write_bytes((S/'cpu'/name).read_bytes())
    previous = json.loads((Q/'cpu-plan.json').read_text())
    controller = json.loads((R/'research/patient-response-code-v1/controller/controller-plan.json').read_text())
    files = ['native_cpu.py','response_codec.py','qwen35_smoke.py','train.original.jsonl.gz','held.original.jsonl.gz']
    plan = dict(CPU_only=True,training_authorized=False,GPU_authorized=False,groups=groups,files={n:ref(B/n) for n in files},tokenizer_files=previous['tokenizer_files'],runtime={**previous['runtime'],'torch':'2.6.0+cu124'},mount_id=controller['mount_id'],remote='patient-response-code/full64-cpu-v1',python=controller['python'],model=controller['model'],env={**controller['env'],'CUDA_VISIBLE_DEVICES':''},cap_seconds=120,term_grace_seconds=5,source=ref(__file__),ancestry=ancestry,prior_assets_identity=controller['payload_files']['model-source-manifest.json'],fresh_full64_required=True,old_adapter_warm_start_forbidden=True,originaldev_gold_read=False,new_quality_authorized=False)
    save(B/'plan.json',plan); save(B/'cpu-plan.json',plan)
    names=files+['plan.json','cpu-plan.json']; file_refs={n:ref(B/n) for n in names}
    with tarfile.open(B/'deployment.tar.gz','x:gz') as archive:
        for n in names: archive.add(B/n,arcname=n,recursive=False)
    save(B/'deployment-manifest.json',dict(archive=ref(B/'deployment.tar.gz'),files=file_refs))
    result=dict(status='PASS_FULL64_SHORT265_FULLTEXT64_RESPONSE_CODE_NO_DEV_GOLD',groups=groups,all_original_users_UTF8_preserved=True,all_train_targets_inverse_exact=True,complete_full_report64_reconstruction_exact=True,train64_originaldev16_disjoint=True,all_candidate_ancestors_exclude_originaldev=True,ancestry=ancestry,old_renderer_called=False,training_prompt_scope='Original frozen short265; NOT expanded全文 training',held_prompt_scope='Existing full-report64; no answer fields',new_NN_calls=0,new_quality_calls=0,originaldev_gold_read=False,source=ref(__file__),seconds=time.monotonic()-started)
    save(B/'structural-result.json',result)
    print(json.dumps(dict(result=ref(B/'structural-result.json'),archive=ref(B/'deployment.tar.gz'),groups={g:{k:v for k,v in r.items() if k not in ['source','output','compressed_original']} for g,r in groups.items()})))

if __name__ == '__main__': main()
