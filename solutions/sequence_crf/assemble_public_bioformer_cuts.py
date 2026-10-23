#!/usr/bin/env python3
"""Fixed .4 and train-OOF-selected .3 decodes, both with identical SFT relations."""
import argparse
import json
from pathlib import Path
import calibrate_public_selector_cut as calibration
base=calibration.base


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();r=Path(a.directory);plan=json.loads((r/'plan.json').read_text())
    cpath=plan['calibration_results'];assert base.sha(cpath['path'])==cpath['sha256']
    result=json.loads(Path(cpath['path']).read_text());assert result['selected_cut']==plan['selected_cut']==.3
    manifest=json.loads((r/'fitted/manifest.json').read_text())
    for name,digest in manifest['outputs'].items():assert base.sha(r/'fitted'/name)==digest
    rows=base.read_jsonl(r/'fitted/augmented.jsonl');scores=base.read_jsonl(r/'fitted/selector-scores.jsonl')
    assert len(rows)==len(scores)==16 and [x['pmc_id'] for x in rows]==[x['pmc_id'] for x in scores]
    association=plan['association_prediction'];assert base.sha(association['path'])==association['sha256']
    originals={d['pmc_id']:d for d in base.read_jsonl(association['path'])};outputs={}
    for cut,name in [(.4,'fixed-point4.jsonl'),(.3,'calibrated-point3.jsonl')]:
        predictions=[]
        for doc,score in zip(rows,scores):
            cs=score['candidates'];ps=[c['selector_probability'] for c in cs]
            entities=calibration.decode(cs,ps,cut)
            if cut==.4:assert entities==doc['entities']
            assert doc['association']==originals[doc['pmc_id']]['association']
            predictions.append(dict(doc,entities=entities))
        (r/name).write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in predictions))
        if cut==.4:assert (r/name).read_bytes()==(r/'fitted/augmented.jsonl').read_bytes()
        outputs[name]={'sha256':base.sha(r/name),'entities':sum(len(d['entities']) for d in predictions),'cut':cut}
    report=dict(source_sha256=base.sha(__file__),decoder_source_sha256=base.sha(calibration.__file__),plan_sha256=base.sha(r/'plan.json'),
        calibration_results_sha256=base.sha(cpath['path']),score_sha256=base.sha(r/'fitted/selector-scores.jsonl'),
        old_association_sha256=association['sha256'],both_association_objects_identical=True,point4_fit_prediction_byteexact=True,
        outputs=outputs,originaldev_labels_read=False)
    (r/'assembly.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
