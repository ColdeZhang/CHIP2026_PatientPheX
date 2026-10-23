"""Discriminating CPU-only contracts; synthetic probabilities are not evidence."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

import features as f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    checks = []

    def passed(name, value):
        assert value, name
        checks.append(name)

    def rejects(name, fn):
        try:
            fn()
        except (ValueError, KeyError):
            checks.append(name)
        else:
            raise AssertionError(name)

    names = {'HP:0000001': 'First preferred name', 'HP:0000002': 'Second preferred name'}
    ka, kb = 'a' * 64, 'b' * 64
    req = [dict(text='surface', hpo_id=hid, name=name, pair_key=key)
           for (hid, name), key in zip(names.items(), [ka, kb])]
    scores = [dict(pair_key=ka, probabilities=[.1, .2, .7]),
              dict(pair_key=kb, probabilities=[.6, .3, .1])]
    obj = f.PairFeatures(names, req, scores)
    candidate = dict(text='surface', identifier='HP:0000001', offset=11, length=7,
                     probability=.8, similarity=.95, public_score=.99)
    old = copy.deepcopy(candidate)
    passed('atomic_class_order', list(obj.features(candidate).values()) == [.1, .2, .7])
    passed('no_candidate_mutation', candidate == old)
    compound = dict(candidate, identifier='HP:0000002;HP:0000001;HP:0000002')
    values = list(obj.features(compound).values())
    passed('compound_unique_atoms_mean', all(math.isclose(x, y, abs_tol=1e-15)
                                              for x, y in zip(values, [.35, .25, .4])))
    passed('compound_order_invariance', obj.features(compound) == obj.features(
        dict(compound, identifier='HP:0000001;HP:0000002')))
    passed('mean_preserves_distribution', math.isclose(sum(values), 1., abs_tol=1e-15))
    passed('NIL_unencoded_zero_sentinel', list(obj.features(dict(candidate, identifier='-1')).values()) == [0., 0., 0.])
    passed('outside_current_branch_zero_sentinel', obj.features(dict(candidate, identifier='HP:9999999')) ==
           dict.fromkeys(f.FEATURES, 0.))
    rejects('mixed_valid_invalid_compound_rejected', lambda: obj.features(dict(candidate, identifier='HP:0000001;-1')))
    rejects('empty_compound_atom_rejected', lambda: obj.features(dict(candidate, identifier='HP:0000001;')))
    rejects('missing_valid_pair_not_zero', lambda: obj.features(dict(candidate, text='SURFACE')))
    rejects('existing_pair_feature_never_overwritten', lambda: obj.features(dict(candidate, phenobert_match=.5)))
    rejects('preferred_name_exact', lambda: f.PairFeatures(names, [dict(req[0], name='synonym')], scores[:1]))
    rejects('duplicate_raw_pair', lambda: f.PairFeatures(names, req + req[:1], scores))
    rejects('duplicate_model_key', lambda: f.PairFeatures(names, req, scores + scores[:1]))
    rejects('missing_model_response', lambda: f.PairFeatures(names, req, scores[:1]))
    rejects('unrequested_model_response', lambda: f.PairFeatures(names, req[:1], scores))
    for name, values in [('nan', [float('nan'), 0., 1.]),
                         ('infinite', [float('inf'), 0., 0.]),
                         ('negative', [-.1, .2, .9]), ('above_one', [1.1, 0., 0.]),
                         ('boolean', [True, 0., 0.]), ('string', ['1', 0., 0.]),
                         ('zero_vector_not_actual_softmax', [0., 0., 0.]),
                         ('wrong_sum', [.2, .3, .4]), ('wrong_class_count', [.5, .5])]:
        rejects('reject_' + name, lambda values=values: f.probabilities(values))
    shared = [dict(req[0]), dict(req[0], text='different original text')]
    shared_obj = f.PairFeatures(names, shared, scores[:1])
    passed('exact_native_repr_forward_reuse_only_via_explicit_request',
           shared_obj.features(candidate) == shared_obj.features(dict(candidate, text='different original text')))
    # Independent OBO fixture exercises root, descendants, out-of-branch and obsolete IDs.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'fixture.obo'
        path.write_text('[Term]\nid: HP:0000118\nname: Phenotypic abnormality\n\n'
                        '[Term]\nid: HP:0000001\nname: current\nis_a: HP:0000118 ! root\n\n'
                        '[Term]\nid: HP:0000002\nname: old\nis_a: HP:0000118 ! root\nis_obsolete: true\n\n'
                        '[Term]\nid: HP:0000003\nname: other branch\n')
        passed('current_branch_includes_root_excludes_obsolete_outside',
               f.preferred_names(path) == {'HP:0000118': 'Phenotypic abnormality', 'HP:0000001': 'current'})
    result = dict(status='PASS_SYNTHETIC_CPU_CONTRACTS_ONLY', count=len(checks), checks=checks,
                  feature_source_sha256=f.sha(f.__file__), checker_sha256=f.sha(__file__),
                  class_order=f.CLASSES, feature_names=f.FEATURES,
                  no_model_loaded=True, no_forward=True, no_fit=True,
                  no_competition_gold_read=True, no_quality_result=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
