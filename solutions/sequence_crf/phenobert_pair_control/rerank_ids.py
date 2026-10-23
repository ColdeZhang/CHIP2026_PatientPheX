"""Keep every original entity boundary; only argmax its same-span ID."""
import copy
import math


def rerank(templates, score_rows):
    assert [x['pmc_id'] for x in templates] == [x['pmc_id'] for x in score_rows]
    outputs, decisions = [], []
    for template, row in zip(templates, score_rows):
        by_span, keys = {}, set()
        for c in row['candidates']:
            key = c['offset'], c['length'], c['identifier']
            assert key not in keys
            keys.add(key)
            p = c['selector_probability']
            assert type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1
            by_span.setdefault(key[:2], []).append(c)
        output = copy.deepcopy(template)
        seen = set()
        for old, new in zip(template['entities'], output['entities']):
            span = old['offset'], old['length']
            assert span not in seen, 'baseline must have unique unchanged boundaries'
            seen.add(span)
            choices = by_span[span]
            original = [c for c in choices if c['identifier'] == old['identifier']]
            assert len(original) == 1, 'original literal ID must remain a candidate'
            assert all(c['text'] == old['text'] for c in choices)
            best = min(choices, key=lambda c: (-c['selector_probability'],
                       c['identifier'] != old['identifier'], c['identifier']))
            new['identifier'] = best['identifier']
            assert {k: v for k, v in new.items() if k != 'identifier'} == {
                k: v for k, v in old.items() if k != 'identifier'}
            decisions.append(dict(pmc_id=template['pmc_id'], offset=span[0], length=span[1],
                                  old_identifier=old['identifier'], new_identifier=best['identifier'],
                                  candidate_count=len(choices), old_probability=original[0]['selector_probability'],
                                  selected_probability=best['selector_probability'],
                                  top_tied_ids=sorted(c['identifier'] for c in choices
                                      if c['selector_probability'] == best['selector_probability'])))
        assert len(output['entities']) == len(template['entities'])
        assert {k: v for k, v in output.items() if k != 'entities'} == {
            k: v for k, v in template.items() if k != 'entities'}
        outputs.append(output)
    return outputs, decisions


def contracts():
    doc = dict(pmc_id='synthetic', patient=[{'patient_id': 'P1'}], association=[None],
               entities=[dict(offset=0, length=2, text='ab', identifier='HP:0000002', note='keep', type='Phenotype')])
    def c(hid, p, offset=0, length=2, text='ab'):
        return dict(identifier=hid, selector_probability=p, offset=offset, length=length, text=text)
    def run(cs): return rerank([doc], [dict(pmc_id='synthetic', candidates=cs)])
    base = c('HP:0000002', .01)
    out, _ = run([base, c('HP:0000001', .02), c('HP:0000003', 1., 4, 2, 'cd')])
    assert out[0]['entities'][0]['identifier'] == 'HP:0000001'  # no .3 cutoff
    assert len(out[0]['entities']) == 1 and out[0]['association'] == [None]
    out, _ = run([c('HP:0000001', .8), c('HP:0000002', .8)])
    assert out == [doc]  # original ID wins a real maximum tie
    out, _ = run([base, c('HP:0000004', .7), c('HP:0000003', .7)])
    assert out[0]['entities'][0]['identifier'] == 'HP:0000003'
    assert rerank([dict(doc, entities=[])], [dict(pmc_id='synthetic', candidates=[])])[0] == [dict(doc, entities=[])]
    for cs in [[c('HP:0000001', .9)], [base, base], [dict(base, text='xy')], [dict(base, selector_probability=float('nan'))]]:
        try: run(cs)
        except (AssertionError, KeyError): pass
        else: raise AssertionError('invalid pool was accepted')
    return {'status': 'PASS_SYNTHETIC_FIXED_SPAN_ID_CONTRACTS', 'cases': 8,
            'no_fit': True, 'no_gold': True, 'no_quality': True}
