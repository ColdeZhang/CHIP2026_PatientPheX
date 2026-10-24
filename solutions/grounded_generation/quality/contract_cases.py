"""Small distinguishing synthetic cases; no real predictions, HPO data, gold or evaluator calls."""
import argparse
from pathlib import Path
from support import (BASE_SCORE, PINS, byte_helpers, canonical_values, endpoint_choice,
                     load_module, now, ref, replace_phenotypes, save)


def contracts(strict_source, byte_helper):
    assert ref(strict_source)['sha256'] == PINS['strict']
    strict = load_module('_grounded_synthetic_strict', ref(strict_source))
    hpo = strict.HPO.__new__(strict.HPO)
    hpo.canonical = {'HP:0000001': 'HP:0000001', 'HP:0000002': 'HP:0000002', 'HP:9999999': 'HP:0000001'}
    assert canonical_values(['HP:0000002;HP:9999999', 'HP:0000001', 'odd; literal phenotype'], hpo) == [
        'HP:0000001', 'HP:0000002', 'odd; literal phenotype']
    rejected = []
    for value in ['-1', 'HP:0000001;-1', 'HP:0000001;MESH:X', 'HP:123', 'HP:1234567']:
        try: canonical_values([value], hpo)
        except ValueError: rejected.append(value)
        else: raise AssertionError(value)
    spans, mask = byte_helpers(ref(byte_helper))
    # No patient field; extra association metadata, escapes, whitespace and entities must survive.
    before = '{ "pmc_id":"synthetic", "entities":[{"text":"é; phenotype", "note":"\\\"phenotype\\\""}], "association" : [{"patient_id":"P1", "tag":{"x":3}, "phenotype": ["HP:0000001"]}, {"phenotype": [], "patient_id":"P2","extra":true}] }\r\n'
    after = replace_phenotypes(before, {'P1': ['HP:0000001', 'odd; literal phenotype'], 'P2': []}, spans, mask)
    assert mask(before) == mask(after) and after.endswith('\r\n')
    assert '"patient"' not in after and '"tag":{"x":3}' in after and '"extra":true' in after
    # Diagnostic is permitted to win even when the primary union hypothesis fails.
    assert endpoint_choice({'primary': {'Score': BASE_SCORE - .01}, 'new-only': {'Score': BASE_SCORE + .01}}) == ('new-only', True)
    assert endpoint_choice({'primary': {'Score': BASE_SCORE + .01}, 'new-only': {'Score': BASE_SCORE + .01}}) == ('primary', True)
    assert endpoint_choice({'primary': {'Score': BASE_SCORE}, 'new-only': {'Score': BASE_SCORE - .01}}) == ('primary', False)
    return dict(status='PASS_SYNTHETIC_BOUNDARIES_ONLY', source=ref(__file__), at_utc=now(),
        strict_source=ref(strict_source), byte_helper=ref(byte_helper), compound_alt_id_overlap_deduplicated=True,
        literal_NIL_semicolon_preserved=True, malformed_or_mixed_identifiers_rejected=rejected,
        baseline_without_patient_field_supported=True, all_other_bytes_and_association_metadata_preserved=True,
        diagnostic_may_win_primary_failure=True, exact_tie_primary=True, strict_improvement_required=True,
        real_input_or_predictions_read=False, gold_read_or_hashed=False, evaluator_calls=0,
        normalization_or_model_calls=0, actual_full_endpoint_contract_not_executed=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict-source', required=True, type=Path)
    parser.add_argument('--byte-helper', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); save(args.output, contracts(args.strict_source, args.byte_helper))
