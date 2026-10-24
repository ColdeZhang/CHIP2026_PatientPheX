"""Bind complete actual positives to the already verified disjoint B legal spaces."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


def ref(path):
    path=Path(path)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):return json.loads(Path(path).read_text())
def checked(item):
    actual=ref(item['path']);assert all(actual[k]==item[k] for k in ('bytes','sha256'));return Path(item['path'])


def main():
    ap=argparse.ArgumentParser()
    for key in ['export-manifest','input-space-proof','output']:ap.add_argument('--'+key,required=True)
    args=ap.parse_args();proof_path=Path(args.input_space_proof)
    assert ref(proof_path)['sha256']=='b68adf7cca53d9a3e203291e326c7809de3ddecf76c6b0ce5e34c14c563bc6da'
    proof=read(proof_path);assert proof['all_inverse_exact'] and proof['original_code_maps_all_exact']
    assert proof['independent_HPO_descendant_alias_graph'] and proof['documents']==100 and proof['pairs']==7147
    for value in proof['references'].values():checked(value)
    expected_path=checked(proof['expected_patient_prompts'])
    expected=[json.loads(line) for line in expected_path.read_text().splitlines()]
    exported=read(args.export_manifest)
    assert exported['source']['sha256']=='caf48c8d14c1496141cc79b6b4af21518ce04267c718e4f0e46499044f235e02'
    checked(exported['source']);checked(exported['plan'])
    for item in exported['references'].values():checked(item)
    assert exported['references']['prompts']['sha256']==ref(expected_path)['sha256']
    assert exported['only_explicit_patient_model_positives'] and exported['no_baseline_fallback']
    assert exported['all_complete_raw_ID_decode'] and exported['raw_first_all_rows'] and exported['codec_replay_exact']
    assert exported['rows']==len(expected)==420 and exported['documents']==100 and exported['patients']==244
    actual=read(checked(exported['legal_spaces']))
    expected_byid={r['row_id']:dict(row_id=r['row_id'],pmc_id=str(r['pmc_id']),patient_ids=r['patient_ids'],code_map=r['code_map']) for r in expected}
    actual_byid={r['row_id']:r for r in actual}
    assert len(actual)==len(actual_byid)==420 and actual_byid==expected_byid
    legal={(str(r['pmc_id']),pid,concept) for r in actual for pid in r['patient_ids'] for concept in r['code_map'].values()}
    positive_rows=read(checked(exported['positive_keys']))
    assert all(isinstance(row,list) and len(row)==3 and all(isinstance(v,str) for v in row) for row in positive_rows)
    positive={tuple(row) for row in positive_rows};assert len(positive)==len(positive_rows)==exported['positive_pairs']
    assert len(legal)==exported['legal_pairs']==7147 and positive<=legal
    public=read(proof['references']['public_spaces']['path'])['documents']
    assert set(public)==set(proof['per_document'])=={r['pmc_id'] for r in actual}
    for doc,summary in proof['per_document'].items():
        assert summary['canonical_intersection']==summary['raw_string_intersection']==0
        assert not {v for did,_,v in positive if did==doc}&set(public[doc]['new_public_canonical_hpo'])
    # Inclusion in the exact certified legal space proves canonical disjointness;
    # no new normalization, filtering, or independent candidate generation occurs.
    report=dict(at_utc=datetime.now(timezone.utc).isoformat(),source=ref(__file__),
        export_manifest=ref(args.export_manifest),input_space_proof=ref(proof_path),
        expected_prompts=ref(expected_path),actual_legal_spaces=exported['legal_spaces'],
        positive_keys=exported['positive_keys'],public_spaces=proof['references']['public_spaces'],
        actual_rows=420,documents=100,patients=244,legal_pairs=7147,positive_pairs=len(positive),
        positive_per_document=dict(sorted(Counter(x[0] for x in positive).items())),
        all_actual_code_maps_and_patients_exact=True,all_actual_positives_in_certified_legal_space=True,
        raw_output_public_intersection=0,canonical_output_public_intersection=0,
        canonical_proof='Actual positives subset of exact B legal spaces already certified canonical-disjoint by pinned independent HPO graph proof',
        literal_text_meanings_preserved=True,no_output_repair_or_filter=True,no_baseline_fallback=True,
        permits_901_equivalent_patient_union=True,no_gold=True,no_score=True,no_NN=True,no_fit=True)
    output=Path(args.output)
    with output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(str(output.parent),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    print(json.dumps(ref(output)))


if __name__=='__main__':main()
