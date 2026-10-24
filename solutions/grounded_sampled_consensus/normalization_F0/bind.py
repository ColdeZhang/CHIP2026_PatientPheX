"""Bind actual all94 fixed-policy queries; no GPU/MCP/fit/held-gold operations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

S = Path(__file__).resolve().parent
N = Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation/normalization')
PRIOR = Path('/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1')
PROJECT = '/home/zhangyuheng/projects/CHIP2026_PatientPheX'
ROOT = PROJECT + '/grounded-sampled-consensus-v1/normalization-F0-v1'
OLD_ROOT = PROJECT + '/grounded-generation-v1/normalization-v1/query-v1'
MOUNT = '5a9955da-5240-41cc-9e52-05d3d8da8a3c'
GPU = 'GPU-126293b9-6677-8464-cee7-01e693efbf29'
SEEDS = [20261024, 20261025]
TRAIN = '38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
INPUT = 'ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37'
HPO = 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b'
PINS = {
    'strict.py': '31d380f582a6ab83e0c73bddbac983c5cf038b250dd2b29355c14f11d8eaebf5',
    'semantic.py': 'e0e015a9b37593f474cb2d053842db74b1992685e972dda3c616b3f3952c9b82',
    'public_bank.py': 'd17acf74a151d8214d9d54af73cedaaadedeb77b956d606fb501e87eb8649061',
    'policy_v1.json': 'b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20',
    'held_query.py': '2d72ada971f59255646384b09ede4400af8c303d5a193f3db38e4303069e71aa',
    'held_driver.py': '3e043fa57e1c23897139028340723b574730df1ef5caf4ddcba7bbdb387d5106',
}


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def checked(item):
    assert ref(item['path']) == item
    return Path(item['path'])


def read(path): return json.loads(Path(path).read_text())
def identity(item): return {k:item[k] for k in ['bytes', 'sha256']}


def gold_guard():
    forbidden = {os.path.realpath(p) for p in [
        '/workspace/patientphex-data/round2-inner4/fold-0/validation16.gold.jsonl',
        '/workspace/patientphex-data/day1/dev16.gold.jsonl']}
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if os.path.realpath(os.fsdecode(args[0])) in forbidden:
                raise PermissionError('Gold cannot be read/hashed by the query binder')
    sys.addaudithook(audit)


def validate_preparation_metadata(prep, queries):
    """Pure structural contract, also used by synthetic CPU checks."""
    assert prep['schema'] == 'grounded-sampled-normalization-preparation-v1'
    assert prep['complete_all94'] is True and prep['rows'] == 94
    assert prep['documents'] == 16 and prep['patients'] == 47 and prep['seeds'] == SEEDS
    assert all(prep[k] is True for k in ['no_gold_fit_or_score','no_unknown_to_NIL','no_threshold_changes'])
    assert prep['shared_queries_only_exact_identical_strings'] is True
    assert prep['train32']['sha256'] == TRAIN and prep['input']['sha256'] == INPUT
    assert prep['hpo']['sha256'] == HPO and prep['policy']['sha256'] == PINS['policy_v1.json']
    assert isinstance(prep['extraction'], dict) and 'extraction_manifest' not in prep
    assert isinstance(queries, list)
    assert all(isinstance(q, dict) and set(q) == {'quote'} and isinstance(q['quote'], str) and q['quote'] for q in queries)
    quotes = [q['quote'] for q in queries]
    assert quotes == sorted(set(quotes))
    assert type(prep['necessary_public_query_count']) is int
    assert len(queries) == prep['necessary_public_query_count']
    assert 20 + .02 * len(queries) <= 180, 'Report larger estimated budget; do not truncate queries'
    return len(queries)


def complete_sampling_release(prep):
    extraction = read(checked(prep['extraction']))
    assert extraction['schema'] == 'grounded-sampled-complete94-extraction-v1'
    assert extraction['complete_all94'] and extraction['rows'] == 94 and extraction['seeds'] == SEEDS
    assert extraction['F0_input'] == prep['input']
    collection = read(checked(extraction['collection']))
    assert collection['mount'] == MOUNT and collection['all_actual_SHA_verified'] and collection['inference_reported_success']
    assert collection['child_exit_before_freeze']['no_running_process']
    wrapper_ref = collection['files']['wrapper-complete.json']; wrapper = read(checked(wrapper_ref))
    terminal_ref = collection['files']['output/complete.json']; terminal = read(checked(terminal_ref))
    assert wrapper == collection['wrapper'] and wrapper['returncode'] == 0 and not wrapper['timed_out']
    assert 'endpoint_error' not in wrapper and wrapper['complete94_actual_members_verified']
    assert wrapper['complete_sha256'] == terminal_ref['sha256']
    assert terminal['status'] == 'COMPLETE_FRESH94_SAMPLED_QUOTE_NO_QUALITY' and terminal['rows'] == 94
    assert wrapper['target_has_no_process'] and 'gpu_release' in wrapper
    release = wrapper['gpu_release']; target = [line.split(',') for line in release['gpu'].splitlines() if GPU in line]
    assert len(target) == 1 and int(target[0][2]) <= 10 and GPU not in release['processes']
    return dict(collection=extraction['collection'], wrapper=wrapper_ref, complete=terminal_ref,
                released_at_utc=release['at_utc'], no_sampling_GPU_process=True)


def main():
    assert __debug__
    ap = argparse.ArgumentParser(); ap.add_argument('--preparation-manifest', required=True)
    ap.add_argument('--release-evidence'); ap.add_argument('--output', default=str(S/'held-query-plan.json'))
    args = ap.parse_args(); gold_guard()
    prep = read(args.preparation_manifest); queries = read(checked(prep['queries']))
    n = validate_preparation_metadata(prep, queries)
    sampling = complete_sampling_release(prep)
    if not n:
        print(json.dumps(dict(no_GPU_query_required=True, preparation=ref(args.preparation_manifest),
                              actual_all94_sampling_release=sampling)))
        return
    host_path = S/'target-assets.json'; host = read(host_path)
    assert host['schema'] == 'grounded-sampled-Sap-existing-assets-v1' and host['mount'] == MOUNT and host['gpu_uuid'] == GPU
    for item in host['proofs'].values(): checked(item)
    oldplan = read(N/'evidence/query-plan.json'); original = PRIOR/'normalization-v1/deployment'
    collection = read(original/'collection.json'); assert collection['all_actual_SHA_verified']
    runtime = read(original/'terminal/actual/encoder-runtime.json')
    complete = read(original/'terminal/actual/complete.json')
    assert complete['status'] == 'COMPLETE_FIXED_PUBLIC_TRAIN_QUERIES' and complete['public_bank_rows'] == 41099 and complete['query_count'] == 624
    sources = {name:ref(N/name) for name in PINS}
    assert {name:item['sha256'] for name,item in sources.items()} == PINS
    for name in ['strict.py','semantic.py','public_bank.py','policy_v1.json']:
        assert sources[name]['sha256'] == oldplan['files'][OLD_ROOT+'/'+name]['sha256']
        assert prep['sources']['reused']['normalization/'+name]['sha256'] == sources[name]['sha256']
    for path,item in oldplan['files'].items():
        if path.startswith(oldplan['model_dir']+'/'):
            assert host['files'][host['model_dir']+'/'+Path(path).name] == item
    assert host['files'][host['hpo']] == oldplan['files'][oldplan['hpo']]
    for name,key in [('public-bank.npy','bank'),('public-labels.json','labels'),('public-bank-manifest.json','bank_manifest')]:
        assert host['files'][host[key]] == complete['assets'][name]
    for key in ['parameters','pooling','max_length']: assert host['expected_encoder_runtime'][key] == runtime[key]
    sources['queries.json'] = prep['queries']
    files = {ROOT+'/'+name:identity(item) for name,item in sources.items()}; files.update(host['files'])
    env = dict(PYTHONPATH=ROOT+':'+host['pythonpath'],PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES=GPU,PATIENTPHEX_DURABILITY_ROOT=PROJECT)
    release_ref = ref(args.release_evidence) if args.release_evidence else sampling['wrapper']
    plan = dict(schema='grounded-fixed-held-public-queries-v1',authorized=True,
        authorization='Root: complete94 sampling and actual GPU0 release first; one necessary fixed-policy train32 public-Sap pass <=180+TERM30; no fit or bank encoding',
        budget_seconds=180,authorized_budget_seconds=180,grace_seconds=30,query_count=n,complete_necessary_queries=True,
        cost_estimate_seconds=20+.02*n,cost_basis='Prior624=15.3473s; conservative20s+.02s/query, actual deadline authoritative',
        remote_root=ROOT,output_dir=ROOT+'/output',python=host['python'],environment=env,gpu_uuid=GPU,storage_parent=PROJECT,
        files=files,payload_sources=sources,reuse_packages=[],reuse_actual_artifacts={},operations_source=ref(S/'remote.py'),
        collection_source=ref(S/'collect.py'),binding_source=ref(__file__),
        input_sources=dict(preparation=ref(args.preparation_manifest),original_public_bank_collection=ref(original/'collection.json'),
            release_evidence=release_ref,original_sampling_collection=sampling['collection'],original_sampling_wrapper=sampling['wrapper'],
            fixed_policy=sources['policy_v1.json'],same_assets_host=ref(host_path)),
        prerequisite_complete94=sampling,host_migration=host,queries=ROOT+'/queries.json',policy=ROOT+'/policy_v1.json',
        hpo=host['hpo'],bank=host['bank'],labels=host['labels'],bank_manifest=host['bank_manifest'],model_dir=host['model_dir'],
        expected_encoder_runtime=host['expected_encoder_runtime'],argv=[host['python'],'-u',ROOT+'/held_query.py','--plan',ROOT+'/held-query-plan.json'])
    target = Path(args.output); assert not target.exists()
    with target.open('x') as stream:
        json.dump(plan,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    parent=target.resolve().parent
    while True:
        fd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        if parent.parent==parent:break
        parent=parent.parent
    print(json.dumps(ref(target)))


if __name__ == '__main__': main()
