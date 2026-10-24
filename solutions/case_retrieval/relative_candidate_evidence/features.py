"""Eighteen input-only relative native-hybrid source features; no fitted scores."""
from collections import defaultdict
import math

SOURCES = ('public', 'bioformer', 'biobert')
GROUPS = ('same_span_other_id', 'same_id_overlapping_span')
FIELDS = ('exists', 'max_score', 'self_minus_max')
FEATURES = tuple(f'competition_{g}_{s}_{f}' for g in GROUPS for s in SOURCES for f in FIELDS)


def relative_features(candidates):
    """Preserve candidate order; inspect only span, literal ID, presence and score.

    Source score is the cached native-hybrid exact-key maximum, not a raw-logit
    posterior. Absent source is explicit zero; malformed or missing keys fail.
    Competitors must carry the particular source. With none, all three features
    are zero. With any, self absent means own score zero and a negative delta.
    """
    spans=defaultdict(list); ids=defaultdict(list); keys=[]; values=[]
    for i,c in enumerate(candidates):
        assert 'selector_probability' not in c, 'FittedSelectorSignalForbidden'
        start,length,identifier=c['offset'],c['length'],c['identifier']
        assert type(start) is int and start>=0 and type(length) is int and length>0
        assert isinstance(identifier,str) and identifier
        key=(start,length,identifier);keys.append(key)
        spans[start,length].append(i);ids[identifier].append(i)
        per_source=[]
        for source in SOURCES:
            present,score=c[source+'_present'],c[source+'_score']
            assert type(present) is bool
            assert type(score) in (int,float,bool) and math.isfinite(float(score)) and 0<=float(score)<=1
            if present:assert type(score) is not bool
            else:assert float(score)==0
            per_source.append((present,float(score)))
        values.append(per_source)
    assert len(set(keys))==len(keys), 'DuplicateLiteralCandidateKey'
    output=[]
    for i,(start,length,identifier) in enumerate(keys):
        same_span=[j for j in spans[start,length] if j!=i and keys[j][2]!=identifier]
        same_id=[j for j in ids[identifier] if j!=i and keys[j][:2]!=(start,length)
                 and start<keys[j][0]+keys[j][1] and keys[j][0]<start+length]
        result={name:0. for name in FEATURES}
        for group,others in zip(GROUPS,(same_span,same_id)):
            for s,source in enumerate(SOURCES):
                scored=[values[j][s][1] for j in others if values[j][s][0]]
                if scored:
                    maximum=max(scored);prefix=f'competition_{group}_{source}_'
                    result[prefix+'exists']=1.
                    result[prefix+'max_score']=maximum
                    result[prefix+'self_minus_max']=values[i][s][1]-maximum
        output.append(result)
    return output
