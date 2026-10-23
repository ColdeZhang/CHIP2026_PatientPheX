#!/usr/bin/env python3
"""Create the predeclared fixed-SFT-association arm, without labels."""
import argparse
import copy
import json
from pathlib import Path
import run as base


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);a=p.parse_args();root=Path(a.directory)
    plan=json.loads((root/'plan.json').read_text())
    best_path=plan['best_prediction']['path'];assert base.sha(best_path)==plan['best_prediction']['sha256']
    assert base.sha(root/'fitted/baseline.jsonl')==plan['baseline_prediction']['sha256']
    manifest=json.loads((root/'fitted/manifest.json').read_text())
    assert manifest['outputs']['augmented.jsonl']==base.sha(root/'fitted/augmented.jsonl')
    prediction=base.read_jsonl(root/'fitted/augmented.jsonl');best={d['pmc_id']:d for d in base.read_jsonl(best_path)}
    assert len(best)==len(prediction)==16 and {d['pmc_id'] for d in prediction}==set(best)
    result=[]
    for doc in prediction:
        new=copy.deepcopy(doc);new['association']=copy.deepcopy(best[doc['pmc_id']]['association'])
        assert new['entities']==doc['entities'] and new['association']==best[doc['pmc_id']]['association']
        result.append(new)
    (root/'preserved-sft.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in result))
    (root/'current-best.jsonl').write_bytes(Path(best_path).read_bytes())
    report=dict(source_sha256=base.sha(__file__),plan_sha256=base.sha(root/'plan.json'),
        new_rules_prediction_sha256=base.sha(root/'fitted/augmented.jsonl'),best_prediction_sha256=base.sha(best_path),
        output_sha256=base.sha(root/'preserved-sft.jsonl'),documents=16,
        entities_identical_to_rule_arm=True,association_objects_identical_to_frozen_best=True,
        gold_read=False,policy='Only replace entities in fixed aa107790 SFT prediction; no relation recomputation or subset filtering.')
    (root/'association-preservation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
