"""Pure-Python set-valued synonym objective specification, not a trainer."""
import json
import math


def partitions(candidate_id_sets, target, known_nonnegative):
    positive = [i for i, ids in enumerate(candidate_id_sets) if target in ids]
    negative = [i for i, ids in enumerate(candidate_id_sets)
                if not (set(ids) & set(known_nonnegative))]
    if not positive:
        raise ValueError('Every target needs at least one public positive alias')
    assert not set(positive) & set(negative)
    return positive, negative


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(x-m) for x in values))


def loss(scores, candidate_id_sets, targets, known_nonnegative):
    assert scores and len(scores) == len(candidate_id_sets)
    assert all(math.isfinite(x) for x in scores)
    result = []
    for target in sorted(targets):
        positive, negative = partitions(candidate_id_sets, target, known_nonnegative)
        result.append(logsumexp([scores[i] for i in positive+negative])
                      - logsumexp([scores[i] for i in positive]))
    return sum(result)/len(result)


def main():
    # A compound annotation means both A and B; B must not be A's negative.
    candidates=[{'A'},{'B'},{'C'},{'A','D'}]
    scores=[2.,3.,1.,0.]
    p,n=partitions(candidates,'A',{'A','B','D'})
    assert p == [0,3] and n == [2]
    p,n=partitions(candidates,'B',{'A','B','D'})
    assert p == [1] and n == [2]
    before=loss(scores,candidates,{'A','B'},{'A','B','D'})
    better=loss([4.,5.,1.,2.],candidates,{'A','B'},{'A','B','D'})
    assert 0 <= better < before
    assert loss([1.,2.],[{'A'},{'A'}],{'A'},{'A'}) == 0
    assert abs(loss([x+1000 for x in scores],candidates,{'A','B'},{'A','B','D'})-before)<1e-12
    # Public same-name ambiguity is masked even if not an annotation target.
    assert partitions([{'A'},{'D'},{'C'}],'A',{'A','D'}) == ([0],[2])
    rejected=False
    try: loss([float('nan')],[{'A'}],{'A'},{'A'})
    except AssertionError: rejected=True
    assert rejected
    print(json.dumps(dict(status='PASS_PURE_MATH_ONLY',checks=6,
        compound_positive_loss=before,improved_positive_loss=better,
        neural_gradient_verified=False,training_performed=False),indent=2))


if __name__ == '__main__': main()
