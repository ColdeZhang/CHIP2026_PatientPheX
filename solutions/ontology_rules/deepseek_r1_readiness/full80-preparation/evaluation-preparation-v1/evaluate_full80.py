#!/usr/bin/env python3
"""Frozen R1 standalone and sole old-SFT32 union: assemble both before F1.

Source preparation alone does not authorize evaluation. --self-test uses only
synthetic fixtures and the frozen standard-library normalized parser.
"""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

SCHEMA = 'patientphex-r1-full80-standalone-union-F1-v1'
ENDPOINTS = ('primary-r1', 'secondary-union')
GOLD_SHA256 = '665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3'
COMPLETE = 'COMPLETE_FIXED80_RESPONSES_2_REUSED_78_NEW_NO_QUALITY_EVALUATION'
SHARED_SHA = {
 'old_baseline_pred':'ffb2c54ac4af0136fe1bb4fa08a34e14d3e0e27c4354a2b9c65aa50ea0f5f379',
 'old_entities':'8acd182b5eee9e1cf56d43f299924de410635d2bc309c8c10bf51be9a59b7068',
 'prompts':'ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe',
 'input':'962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415',
 'old_baseline_metrics':'7e36dd93a436bdaadd4b0527b1bb388d6e438d8f4bcf8748d00020a3a1c4f477',
 'hpo':'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',
 'evaluator':'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781',
 'validator':'b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f',
 'normalized_final':'9c7cd604cae230f8a77bf0be3fd7edf2a3893e35c584de38edf8724c11c43efe',
 'strict_final':'20943d8681b4cf81aee0465b0f217d3491460a311be00f3bbc2818b721013fad',
 'input_audit':'36003a4e5464fbf75325a1af8fca4370d1363f2671dd10dc9a5757cdaa557229',
}
DYNAMIC_NAMES = {'inference_source','inference_launcher','inference_plan','inference_result',
                 'responses','wrapper_result','cached_row0','cached_row73'}
POLICY = {
 'primary':'R1 normalized-only provided relations; invalid whole rows and missing CIDs abstain without fallback.',
 'secondary':'Only literal per-document per-patient set union of old-SFT32 and primary R1 relations.',
 'normalizer_sha256':SHARED_SHA['normalized_final'],
 'selection_comparator':'Cached old-SFT32 baseline; no old reevaluation.',
 'retention':'relation micF1+macF1 strictly increases AND macF1 does not decrease',
 'evaluation':'Both complete predictions frozen first; each exactly one unchanged v2 all/text call.',
 'nonrelation':'Old baseline bytes outside each top-level association value remain exactly unchanged.',
 'scope':'Known F1 development only; no F0/originaldev/official or independent-generalization claim.',
 'automatic_migration':False, 'parameter_or_threshold_search':False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path':str(path), 'bytes':path.stat().st_size, 'sha256':sha(path)}


def checked(record):
    require(set(record) == {'path','bytes','sha256'}, 'Require immutable three-field ref')
    path = Path(record['path'])
    require(path.is_absolute() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
            'Artifact SHA/size mismatch: ' + str(path))
    return path


def same_bytes(actual, recorded):
    require(actual['bytes'] == recorded['bytes'] and actual['sha256'] == recorded['sha256'],
            'Recovered/original artifact bytes differ')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def canonical_item_sha(item):
    return hashlib.sha256(json.dumps(item,ensure_ascii=False,sort_keys=True,
                                    separators=(',',':'),allow_nan=False).encode()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'Cannot load frozen source')
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def load_normalizer(normalized, strict):
    require(sha(normalized) == SHARED_SHA['normalized_final'] and sha(strict) == SHARED_SHA['strict_final'],
            'Frozen normalized/unique-key codec differs')
    if 'strict_final' in sys.modules:
        require(sha(sys.modules['strict_final'].__file__) == SHARED_SHA['strict_final'], 'Conflicting strict module')
    else:
        module('strict_final', strict)
    return module('frozen_r1_normalized', normalized)


# These five functions are copied byte-for-byte from the frozen MedGemma entry
# SHA22aec1f43fc6c5498404bd7e9cfb2c599216a6d94b8d5cc68e3f31752081744b.
# Its internal med variable/count names are only local names; no Med parser runs.
def association_span(line):
    """Locate the one top-level value without reserializing any other JSON bytes."""
    decoder, position, names, found = json.JSONDecoder(), 0, set(), None
    def whitespace(index):
        while index < len(line) and line[index].isspace():
            index += 1
        return index
    position = whitespace(position)
    require(line[position:position + 1] == '{', 'Require JSON object line')
    position += 1
    while True:
        position = whitespace(position)
        if line[position:position + 1] == '}':
            position += 1
            break
        key, position = decoder.raw_decode(line, position)
        require(isinstance(key, str) and key not in names, 'Duplicate/nonstring top-level key')
        names.add(key)
        position = whitespace(position)
        require(line[position:position + 1] == ':', 'Missing object colon')
        begin = whitespace(position + 1)
        value, end = decoder.raw_decode(line, begin)
        if key == 'association':
            require(isinstance(value, list), 'Association must be a list')
            found = (begin, end)
        position = whitespace(end)
        if line[position:position + 1] == ',':
            position += 1
            continue
        require(line[position:position + 1] == '}', 'Invalid object separator')
        position += 1
        break
    require(not line[position:].strip() and found is not None, 'Missing association/trailing JSON')
    return found

def nonrelation_bytes(lines):
    output = []
    for line in lines:
        begin, end = association_span(line)
        output.append((line[:begin] + '[]' + line[end:]).encode('utf-8'))
    return b''.join(output)

def relation_map(document):
    result = {}
    for association in document['association']:
        require(set(association) == {'patient_id', 'phenotype'}, 'Unexpected association fields')
        patient = str(association['patient_id'])
        phenotypes = association['phenotype']
        require(patient not in result and isinstance(phenotypes, list)
                and all(isinstance(value, str) for value in phenotypes)
                and len(phenotypes) == len(set(phenotypes)), 'Duplicate/invalid patient or phenotype')
        result[patient] = set(phenotypes)
    return result

def assemble_two(old_lines, med_documents):
    baseline = [json.loads(line) for line in old_lines]
    require(len(baseline) == len(med_documents), 'Document coverage differs')
    require(len({str(d['pmc_id']) for d in baseline}) == len(baseline), 'Duplicate baseline document')
    primary, union = [], []
    counts = {'old': 0, 'med': 0, 'intersection': 0, 'union': 0}
    for raw, old, med in zip(old_lines, baseline, med_documents):
        require({k: v for k, v in old.items() if k != 'association'}
                == {k: v for k, v in med.items() if k != 'association'}, 'Nonrelation fields/document order changed')
        old_map, med_map = relation_map(old), relation_map(med)
        require(set(old_map) == set(med_map), 'Patient coverage differs')
        combined = copy.deepcopy(old['association'])
        for association in combined:
            patient = str(association['patient_id'])
            left, right = old_map[patient], med_map[patient]
            association['phenotype'] = sorted(left | right)
            counts['old'] += len(left)
            counts['med'] += len(right)
            counts['intersection'] += len(left & right)
            counts['union'] += len(left | right)
        begin, end = association_span(raw)
        primary.append(raw[:begin] + json.dumps(med['association'], ensure_ascii=False) + raw[end:])
        union.append(raw[:begin] + json.dumps(combined, ensure_ascii=False) + raw[end:])
    unchanged = nonrelation_bytes(old_lines)
    require(nonrelation_bytes(primary) == nonrelation_bytes(union) == unchanged, 'Nonrelation byte drift')
    return primary, union, counts

def retain(candidate, baseline):
    values = [candidate[k] for k in ('micF1', 'macF1')] + [baseline[k] for k in ('micF1', 'macF1')]
    require(all(math.isfinite(value) for value in values), 'Nonfinite relation metric')
    return candidate['micF1'] + candidate['macF1'] > baseline['micF1'] + baseline['macF1'] and candidate['macF1'] >= baseline['macF1']



def parse_evidence(prompt, item, codec):
    """Only the fixed normalized codec decides evidence; never strict/fallback."""
    try:
        evidence = codec.parse_normalized(item['final_text_excluding_terminal_EOS'],
            eos_observed=item['native_EOS_observed'],
            stopped_at_cap=item['stopped_at_token_cap'] or item['stopped_at_wall_budget'],
            code_map=prompt['code_map'], patient_ids=prompt['patient_ids'])
        valid = True
    except (ValueError, TypeError):
        evidence, valid = None, False
    require(type(item['normalized_valid']) is bool and item['normalized_valid'] is valid,
            'Recomputed fixed normalized verdict differs from raw receipt')
    if valid:
        require(evidence == item.get('normalized_evidence'), 'Normalized evidence differs from immutable raw receipt')
        require(evidence['missing_CIDs_are_negative'] is False and
                evidence['semantic_answer_repair_performed'] is False, 'Unexpected negative fill or semantic repair')
    else:
        require('normalized_evidence' not in item, 'Invalid row retains stale evidence')
    return evidence


def provided_relations(prompts, evidence):
    require(len(prompts) == len(evidence), 'Cannot silently zip incomplete parsed evidence')
    relations = defaultdict(lambda: defaultdict(set))
    statistics = Counter()
    for prompt, parsed in zip(prompts, evidence):
        pmc = prompt['pmc_id']
        if parsed is None:
            statistics['invalid_rows'] += 1
            statistics['abstained_invalid_row_CIDs'] += len(prompt['code_map'])
            continue
        statistics['valid_rows'] += 1
        statistics['abstained_missing_CIDs'] += len(parsed['abstained_CIDs'])
        for cid, patients in parsed['provided_mapping'].items():
            require(cid in prompt['code_map'] and all(type(pid) is str and pid in prompt['patient_ids'] for pid in patients),
                    'Codec output is outside this original row map/patient list')
            statistics['provided_CIDs'] += 1
            statistics['explicit_empty_provided_CIDs'] += not patients
            concept = prompt['code_map'][cid]
            require(type(concept) is str, 'Original concept must remain a string')
            for patient in patients:
                relations[pmc][patient].add(concept)
    return relations, dict(statistics)


def make_primary(old_lines, inputs, prompts, evidence):
    baseline = [json.loads(line) for line in old_lines]
    require([str(doc['pmc_id']) for doc in baseline] == [str(doc['pmc_id']) for doc in inputs],
            'Original input/baseline order differs')
    require(len(baseline) == len({str(doc['pmc_id']) for doc in baseline}), 'Duplicate original baseline document')
    allowed = {}
    for doc in inputs:
        require('entities' not in doc and 'association' not in doc, 'Input file contains labels')
        patients = [patient['patient_id'] for patient in doc['patient']]
        require(all(type(pid) is str for pid in patients) and len(patients) == len(set(patients)), 'Ambiguous original patient IDs')
        allowed[str(doc['pmc_id'])] = patients
    doc_patients, concepts = {}, set()
    for prompt in prompts:
        pmc = prompt['pmc_id']
        require(pmc in allowed and set(prompt['patient_ids']) == set(allowed[pmc]), 'Prompt/input patient set differs')
        require(pmc not in doc_patients or doc_patients[pmc] == prompt['patient_ids'], 'Patient order differs across chunks')
        doc_patients[pmc] = prompt['patient_ids']
        require(list(prompt['code_map']) == ['C%03d' % i for i in range(len(prompt['code_map']))], 'CID sequence differs')
        for concept in prompt['code_map'].values():
            require((pmc,concept) not in concepts, 'Duplicate original document concept')
            concepts.add((pmc,concept))
    require(set(doc_patients) == set(allowed), 'Prompt document coverage differs')
    relations, stats = provided_relations(prompts, evidence)
    primary = []
    for old in baseline:
        pmc = str(old['pmc_id'])
        require(all(type(item['patient_id']) is str for item in old['association']), 'Baseline coerces patient identity')
        require(set(relation_map(old)) == set(allowed[pmc]), 'Baseline patient set differs')
        document = copy.deepcopy(old)
        document['association'] = [{'patient_id':item['patient_id'],
                                   'phenotype':sorted(relations[pmc][item['patient_id']])}
                                  for item in old['association']]
        primary.append(document)
    return primary, stats


def reference_gate(plan, plan_path, responses_path, result_path):
    require(plan['execution_ready'] is True and plan['schema'] == SCHEMA and plan['policy'] == POLICY,
            'Evaluation not authorized or fixed endpoint policy differs')
    require(plan['control_sha256'] == sha(__file__) and plan['historical_F1_gold_sha256'] == GOLD_SHA256,
            'Evaluation source/gold metadata differs')
    common, endpoints = plan['references'], plan['endpoint_references']
    require(set(common) == set(SHARED_SHA) and set(endpoints) == DYNAMIC_NAMES, 'Reference whitelist differs')
    for name, expected in SHARED_SHA.items():
        require(common[name]['sha256'] == expected, 'Frozen shared reference differs: ' + name)
    gold = Path(plan['gold_path']).resolve()
    records = list(common.values()) + list(endpoints.values())
    for record in records:
        require(Path(record['path']).resolve() != gold, 'Gold cannot be a prechecked reference')
        if Path(record['path']).exists() and gold.exists():
            require(not os.path.samefile(record['path'],gold), 'Gold alias cannot be prechecked')
        checked(record)
    require(Path(responses_path).resolve() == checked(endpoints['responses']).resolve() and
            Path(result_path).resolve() == checked(endpoints['inference_result']).resolve(), 'CLI artifacts differ from plan')
    return common, endpoints, records


def verify_inference(common, endpoints, prompts, audit):
    infer, result, wrapper = (read(endpoints[name]['path']) for name in ('inference_plan','inference_result','wrapper_result'))
    require(result['status'] == COMPLETE and result['quality'] is False and
            result['new_generations'] == 78 and result['reused_generations'] == 2 and
            result['original80_target'] == len(result['rows']) == 80 and result['original_order_preserved'] is True,
            'Require complete 2cached+78fresh inference; incomplete runs cannot enter quality')
    same_bytes(endpoints['inference_source'],result['source'])
    same_bytes(endpoints['inference_plan'],result['plan'])
    same_bytes(endpoints['responses'],result['response_bundle'])
    require(wrapper['status'] == 'owned_probe_process_completed' and wrapper['returncode'] == 0 and
            wrapper['fixed80_2cached_78fresh'] is True and wrapper['no_gold_quality'] is True and
            wrapper['cap_seconds'] == 10800 and wrapper['TERM_grace_seconds'] == 30 and
            wrapper['result_sha256'] == endpoints['inference_result']['sha256'] and wrapper['result_status'] == COMPLETE and
            wrapper['plan_sha256'] == endpoints['inference_plan']['sha256'] and
            wrapper['source_sha256'] == endpoints['inference_launcher']['sha256'], 'Incomplete/timed-out wrapper or source binding')
    require(wrapper['target_UUID_has_no_compute_process'] is True and 'actual_release' in wrapper,
            'Missing completed GPU release receipt')
    require([part.strip() for part in wrapper['actual_release']['gpu'].split(',')][1] == infer['gpu_uuid'],
            'Release GPU UUID differs from the frozen inference target')
    require(not any(line.split(',')[0].strip() == infer['gpu_uuid'] for line in wrapper['actual_release']['compute_apps'].splitlines()),
            'Release flag differs from recorded compute processes')
    require(infer['execution_ready'] is True and infer['process_budget_seconds'] == 10800 and
            infer['TERM_grace_seconds'] == 30 and infer['max_new_tokens'] == 4096 and infer['seed'] == 20261024 and
            infer['temperature'] == .6 and infer['top_p'] == .95 and infer['top_k'] == 0 and
            infer['RNG_policy'] == 'fresh_same_seed_per_row_not_continuous_RNG', 'Fixed inference parameters differ')
    require(infer['prompts_uncompressed_sha256'] == common['prompts']['sha256'], 'Original80 prompt pin differs')
    same_bytes(common['input_audit'],infer['all80_audit'])
    require(infer['row_ids'] == [row['row_id'] for row in prompts] and
            infer['new_indices'] == [i for i in range(80) if i not in (0,73)] and
            set(infer['cached_rows']) == {'0','73'}, 'Frozen inference row order/reuse split differs')
    source_dir = Path(endpoints['inference_source']['path']).resolve().parent
    require(source_dir / 'infer_full80.py' == Path(endpoints['inference_source']['path']).resolve() and
            source_dir / 'launch_full80.py' == Path(endpoints['inference_launcher']['path']).resolve(), 'Inference source directory differs')
    source_refs = {}
    for name, expected in infer['sources'].items():
        require(Path(name).name == name, 'Nonlocal source filename')
        actual = ref(source_dir/name)
        require(actual['sha256'] == expected, 'Frozen inference dependency source changed: '+name)
        source_refs[name] = actual
    require(source_refs['infer_full80.py']['sha256'] == endpoints['inference_source']['sha256'] and
            source_refs['launch_full80.py']['sha256'] == endpoints['inference_launcher']['sha256'] and
            source_refs['normalized_final.py']['sha256'] == SHARED_SHA['normalized_final'] and
            source_refs['strict_final.py']['sha256'] == SHARED_SHA['strict_final'], 'Inference parser/source recipe changed')
    require(result['parameters']['count'] == 8190735360 and result['parameters']['tensors'] == 399 and
            result['parameters']['all_actual_shapes_dtype_finite'] is True and
            not any(result['loading_information'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),
            'Native full-model loading receipt incomplete')
    require(result['runtime']['precision'] == 'native_BF16_SDPA_no_quantization' and
            result['runtime']['TF32_matmul'] is False and result['runtime']['TF32_cudnn'] is False and
            result['actual_rotary']['attention_scaling'] == 1.0, 'Actual native runtime differs')
    data = gzip.decompress(checked(endpoints['responses']).read_bytes())
    require(hashlib.sha256(data).hexdigest() == result['response_uncompressed_sha256'], 'Full uncompressed response hash differs')
    raw = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    require(len(raw) == len(prompts) == len(audit) == 80 and len({item['row_id'] for item in raw}) == 80,
            'Full80 response coverage differs')
    codec = load_normalizer(common['normalized_final']['path'],common['strict_final']['path'])
    evidence, reused = [], []
    fields = ['row_id','original_index','fresh_NN','input_tokens','generated_tokens','native_EOS_observed',
              'stopped_at_token_cap','stopped_at_wall_budget','generation_seconds','strict_final_valid',
              'legal_sparse_valid','normalized_valid']
    for i,(prompt,item,receipt,cpu) in enumerate(zip(prompts,raw,result['rows'],audit)):
        require(i == item['original_index'] == receipt['original_index'] == cpu['index'] and
                item['row_id'] == prompt['row_id'] == cpu['row_id'] and prompt['pmc_id'] == cpu['pmc_id'], 'Response/CPU row identity differs')
        require(hashlib.sha256(json.dumps(prompt,ensure_ascii=False,separators=(',',':')).encode()).hexdigest() == cpu['original_row_sha256'],
                'CPU audit does not bind this original prompt row')
        require(item['input_tokens'] == cpu['native_input_tokens'] and item['quality_evaluated'] is False and
                item['stopped_at_wall_budget'] is False, 'Changed input or global timeout cannot enter quality')
        require(all(item[key] == receipt[key] for key in fields) and
                canonical_item_sha(item) == receipt['canonical_item_sha256'], 'Full raw item is not bound to original row receipt')
        require(Path(receipt['path']).name == 'row%d.json'%i and receipt['bytes'] > 0 and len(receipt['sha256']) == 64,
                'Original per-row file reference missing')
        ids = item['generated_ids']
        require(len(ids) == item['generated_tokens'] <= 4096 and all(type(v) is int and 0 <= v < 151936 for v in ids), 'Generated-ID vector/count invalid')
        require(type(item['native_EOS_observed']) is bool and
                item['native_EOS_observed'] == bool(ids and ids[-1] == 151645) and
                item['stopped_at_token_cap'] == (not item['native_EOS_observed'] and len(ids) >= 4096), 'EOS/cap evidence differs')
        for flag in ('strict_final_valid','legal_sparse_valid','normalized_valid','fresh_NN','stopped_at_token_cap'):
            require(type(item[flag]) is bool, 'Invalid row flag type')
        if i in (0,73):
            require(item['fresh_NN'] is False and item['reused_cached_reference'] == infer['cached_rows'][str(i)], 'Reused row identity differs')
            cached_ref = endpoints['cached_row'+str(i)]
            same_bytes(cached_ref,infer['cached_rows'][str(i)])
            cached = read(cached_ref['path'])
            exclude = {'coverage','original_index','reused_cached_reference','fresh_NN'}
            require(canonical_item_sha({k:item[k] for k in cached if k not in exclude}) ==
                    canonical_item_sha({k:v for k,v in cached.items() if k not in exclude}), 'Reused raw generation was modified')
            reused.append(i)
        else:
            require(item['fresh_NN'] is True and item['seed'] == 20261024 and
                    'reused_cached_reference' not in item and item['input_ids_sha256'] == cpu['native_ids_sha256'],
                    'Fresh row used a cache or different input/seed')
        parsed = parse_evidence(prompt,item,codec)
        cover = item['coverage']
        require(cover['missing_CIDs_are_negative'] is False and cover['fallback_used'] is False and
                cover['invalid_whole_row_abstention'] is (parsed is None), 'Coverage/fallback policy differs')
        require(cover['provided_CIDs'] == (list(parsed['provided_mapping']) if parsed else []) and
                cover['abstained_CIDs'] == (parsed['abstained_CIDs'] if parsed else list(prompt['code_map'])), 'Actual coverage bookkeeping differs')
        evidence.append(parsed)
    require(reused == [0,73] and sum(cpu['native_input_tokens'] for cpu in audit) == 249937 and
            audit[0]['native_input_tokens'] == 2014 and audit[73]['native_input_tokens'] == 5689, 'Reuse/input-token inventory differs')
    actual = {'rows':80,'new_generations':78,'reused_generations':2,
              'valid_rows':sum(value is not None for value in evidence),
              'abstained_invalid_rows':sum(value is None for value in evidence),
              'token_capped_rows':sum(item['stopped_at_token_cap'] for item in raw),
              'provided_CIDs':sum(len(value['provided_mapping']) for value in evidence if value is not None),
              'missing_CIDs_are_negative':False,'invalid_rows_have_no_fallback':True}
    require(result['coverage'] == actual, 'Complete inference coverage differs from actual raw bundle')
    return evidence, actual, source_refs


def delta(current, baseline):
    return {key:current[key]-baseline[key] for key in ('micF1','macF1','menF1','docF1','Score')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan');parser.add_argument('--responses');parser.add_argument('--inference-result')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.self_test:
        require(not any((args.plan,args.responses,args.inference_result)), 'Synthetic mode cannot consume real artifacts')
        print(json.dumps(self_test()));return
    require(all((args.plan,args.responses,args.inference_result)), 'Require --plan --responses --inference-result')
    started = time.monotonic();plan_path=Path(args.plan).resolve();plan_ref=ref(plan_path);plan=read(plan_path)
    common,endpoints,records=reference_gate(plan,plan_path,args.responses,args.inference_result)
    prompts=rows(common['prompts']['path']);audit=read(common['input_audit']['path']);inputs=rows(common['input']['path'])
    require(len(prompts)==80 and len({row['pmc_id'] for row in prompts})==16 and
            sum(len(row['code_map']) for row in prompts)==587 and
            sum(len(row['code_map'])*len(row['patient_ids']) for row in prompts)==1557, 'Original F1 input universe differs')
    evidence,coverage,source_refs=verify_inference(common,endpoints,prompts,audit)
    old_lines=Path(common['old_baseline_pred']['path']).read_bytes().decode('utf-8').splitlines(keepends=True)
    primary_documents,parse_stats=make_primary(old_lines,inputs,prompts,evidence)
    require(sum(parse_stats.get(key,0) for key in ('provided_CIDs','abstained_invalid_row_CIDs','abstained_missing_CIDs')) == 587,
            'Every original CID must be provided or explicitly abstained exactly once')
    require(len(primary_documents)==16 and sum(len(d['entities']) for d in primary_documents)==1189,
            'Frozen full-document/entity scope differs')
    primary,union,counts=assemble_two(old_lines,primary_documents);counts['r1']=counts.pop('med')
    directory=Path(plan['output_dir']).resolve();directory.mkdir(parents=True,exist_ok=False)
    frozen={}
    for name,lines in zip(ENDPOINTS,(primary,union)):
        target=directory/name;target.mkdir()
        path=target/'pred.jsonl'
        with path.open('xb') as handle:handle.write(''.join(lines).encode('utf-8'));handle.flush();os.fsync(handle.fileno())
        frozen[name]={'prediction':ref(path)}
    for record in records:checked(record)
    require(sha(plan_path)==plan_ref['sha256'], 'Plan changed during assembly')
    freeze_path=directory/'paired-predictions-freeze.json'
    save(freeze_path,{'status':'both_predeclared_complete_full_predictions_frozen_before_gold','at_utc':datetime.now(timezone.utc).isoformat(),
        'plan':plan_ref,'endpoints':frozen,'inference_result':endpoints['inference_result'],'responses':endpoints['responses'],
        'inference_sources':source_refs,'old':common['old_baseline_pred'],'coverage':coverage,'parse':parse_stats,
        'union_literal_counts':counts,'documents':16,'entities':1189,'nonrelation_bytes_exact':True,
        'masked_nonrelation_sha256':hashlib.sha256(nonrelation_bytes(old_lines)).hexdigest(),
        'row_binding':'Full gzip SHA and uncompressed SHA; every original-index row canonical_item_sha256 matches inference receipt. Original standalone row-file byte refs are preserved upstream; not falsely claimed individually downloaded.',
        'gold_read':False,'new_quality_calls':0})
    freeze_ref=ref(freeze_path)
    for value in frozen.values():checked(value['prediction'])
    checked(freeze_ref)
    save(directory/'evaluation-started.json',{'freeze':freeze_ref,'fixed_new_v2_calls':2,'cached_old_only':True})
    # Both complete immutable outputs and their SHA checks precede evaluator import and the FIRST gold byte.
    evaluator=module('r1_fixed_evaluate',common['evaluator']['path'])
    validator=module('r1_fixed_validate',common['validator']['path'])
    require(evaluator.VERSION=='patientphex-public-approx-v2','Evaluator version changed')
    hpo=validator.read_hpo(common['hpo']['path'])
    predictions={}
    for name in ENDPOINTS:
        predictions[name]=rows(checked(frozen[name]['prediction']))
        validation=validator.validate_predictions(inputs,predictions[name],hpo)
        save(directory/name/'validation.json',validation)
        require(validation['valid'],'Invalid full prediction: '+name)
    require(sha(plan['gold_path'])==GOLD_SHA256,'Historical F1 gold bytes changed')
    truth=rows(plan['gold_path'])
    require(len(truth)==16 and [str(doc['pmc_id']) for doc in truth]==[str(doc['pmc_id']) for doc in inputs], 'Historical F1 identity/order changed')
    old=read(common['old_baseline_metrics']['path']);metrics={}
    for name in ENDPOINTS:
        checked(frozen[name]['prediction'])
        metrics[name]=evaluator.evaluate_report(truth,predictions[name],'all','text')
        require(metrics[name]['documents']==16 and metrics[name]['patients']==39 and
                all(metrics[name]['metrics'][key]==old['metrics'][key] for key in ('menF1','docF1')), 'Nonrelation metrics/F1 scope drift')
        save(directory/name/'metrics.json',metrics[name])
    for record in records:checked(record)
    checked(freeze_ref)
    for value in frozen.values():checked(value['prediction'])
    require(sha(plan_path)==plan_ref['sha256'],'Plan changed during fixed endpoint')
    summary={'status':'complete_predeclared_R1_standalone_and_oldSFT32_union_F1_development','policy':POLICY,
        'paired_freeze':freeze_ref,'old_cached_metrics':common['old_baseline_metrics'],'old_metrics':old['metrics'],
        'endpoints':{name:{'metrics':metrics[name]['metrics'],'counts':metrics[name]['counts'],
             'metrics_file':ref(directory/name/'metrics.json'),'format_file':ref(directory/name/'validation.json'),
             'delta_vs_old':delta(metrics[name]['metrics'],old['metrics']),
             'retain_vs_old_advisory':retain(metrics[name]['metrics'],old['metrics'])} for name in ENDPOINTS},
        'union_minus_r1_report_only':delta(metrics[ENDPOINTS[1]]['metrics'],metrics[ENDPOINTS[0]]['metrics']),
        'new_v2_calls':2,'automatic_migration':False,'elapsed_CPU_seconds':time.monotonic()-started}
    save(directory/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'summary':ref(directory/'summary.json')}))


def self_test():
    base=Path(__file__).resolve().parent
    codec=load_normalizer(base/'normalized_final.py',base/'strict_final.py')
    prompt={'row_id':'S:relations:0','pmc_id':'S','code_map':{'C000':'HP:0000001','C001':'literal concept'},'patient_ids':['P1','OII-1'],'messages':[]}
    def fixture(text,eos=True,cap=False):
        item={'final_text_excluding_terminal_EOS':text,'native_EOS_observed':eos,'stopped_at_token_cap':cap,'stopped_at_wall_budget':False}
        try:item['normalized_evidence']=codec.parse_normalized(text,eos_observed=eos,stopped_at_cap=cap,code_map=prompt['code_map'],patient_ids=prompt['patient_ids']);item['normalized_valid']=True
        except (ValueError,TypeError):item['normalized_valid']=False
        return item
    sparse=parse_evidence(prompt,fixture('```json\n{"C000":[1]}\n```'),codec)
    require(sparse['provided_mapping']=={'C000':['P1']} and sparse['abstained_CIDs']==['C001'],'Sparse alias/no-fill fixture failed')
    invalid=parse_evidence(prompt,fixture('{"C000":["P1"],"C001":[999]}'),codec)
    require(invalid is None,'Invalid whole row retained a valid fragment')
    capped=parse_evidence(prompt,fixture('{"C000":["P1"]}',False,True),codec)
    require(capped is None,'Truncation received fallback')
    empty=parse_evidence(prompt,fixture('{}'),codec)
    require(empty['provided_mapping']=={} and empty['abstained_CIDs']==['C000','C001'],'Missing CID was filled')
    literal=parse_evidence(prompt,fixture('{"C001":["OII-1"]}'),codec)
    rel,stats=provided_relations([prompt],[literal]);require(rel['S']['OII-1']=={'literal concept'},'Literal concept/patient identity changed')
    raw=' {"pmc_id":"S", "pmid": null, "entities": [{"text":"\\u4e2d"}], "association" : [{"patient_id":"P1","phenotype":["OLD"]},{"patient_id":"OII-1","phenotype":[]}], "extra":"é"}\r\n'
    inputs=[{'pmc_id':'S','pmid':None,'patient':[{'patient_id':'P1'},{'patient_id':'OII-1'}],'full_text':[]}]
    docs,_=make_primary([raw],inputs,[prompt],[sparse]);primary,union,counts=assemble_two([raw],docs)
    require(nonrelation_bytes(primary)==nonrelation_bytes(union)==nonrelation_bytes([raw]),'Nonassociation byte drift')
    require(relation_map(json.loads(primary[0]))=={'P1':{'HP:0000001'},'OII-1':set()},'Standalone filled missing evidence')
    require(relation_map(json.loads(union[0]))=={'P1':{'OLD','HP:0000001'},'OII-1':set()},'Union not literal')
    invalid_docs,_=make_primary([raw],inputs,[prompt],[invalid]);require(all(not x['phenotype'] for x in invalid_docs[0]['association']),'Invalid row used old-baseline fallback')
    for broken in ['{"association":[],"association":[]}', '{"a":1}', '{"association":{}}']:
        try:association_span(broken)
        except ValueError:pass
        else:raise AssertionError('Invalid association structure accepted')
    require(retain({'micF1':.6,'macF1':.5},{'micF1':.5,'macF1':.5}) and
            not retain({'micF1':.7,'macF1':.4},{'micF1':.5,'macF1':.5}),'Retention policy changed')
    return {'status':'PASS_pure_synthetic_bytes_no_fill_invalid_union_fixtures','real_artifacts_or_gold_read':False,
            'model_tokenizer_or_quality_executed':False,'normalized_source_sha256':SHARED_SHA['normalized_final']}


if __name__=='__main__':
    main()
