# B-only coreference markers: old80 / old SFT32 F1 control

REJECT under the precommitted training-side F1 rule. The rule requires Score to strictly increase and macro F1 not to decrease. This is repeatedly observed training-side F1, not original-dev or official/generalization evidence.

| Metric | Cached original80 SFT32 | B markers | Difference |
|---|---:|---:|---:|
| Score | 0.660942003 | 0.660756991 | -0.000185012 |
| micF1 | 0.686600221 | 0.666666667 | -0.019933555 |
| macF1 | 0.542228853 | 0.561422361 | +0.019193508 |
| menF1 | 0.667191188 | 0.667191188 | +0.000000000 |
| docF1 | 0.747747748 | 0.747747748 | +0.000000000 |

New relation counts: 301 TP / 82 FP / 219 FN. Added relations versus cached baseline: 29, including 11 correct and 18 incorrect. Newly introduced wrong supplied-patient assignments: **14**; added concepts associated with no supplied patient in gold: 4. Removed correct/incorrect relations: 20/9. Total wrong supplied-patient assignments before/after: 28/35.

Exactly 80 requests, 587 candidate codes and 1,557 patient/code pairs used the old fixed SFT32 adapter (weight SHA4f4815b6…, config96564441…) and unchanged Qwen3.5 runner33a5a497…/parser. All 1,189 entities and every non-association field matched the baseline. Full80 raw responses and complete16-document predictions were frozen before the single F1 score; no union, intersection, retraining, other mapping arm or original-dev access occurred.

Only B's previously frozen identity candidates were inserted as [COREF patientID: text] tags inside already selected old contexts, plus one fixed system caution. All80 rows restore exactly after removing those tags/caution. Original patient markers, global anchors, candidate/CID/patient fields and all982 selected contexts/order were preserved. Markers appeared in 67/80 rows and 424/982 contexts, covering 286/392 F1 B candidates (1645 repeated marker occurrences). Every marker binds to source input SHA, native output/cluster, patient ID and original global coordinates. No aliases from gold were used.

Actual cached tokenizer audit matched all original token counts. New max/total input tokens: 5895/264626 (old5691/250696); cap14,000, outputcap1,600, greedy/no-thinking, no truncation. One A800 GPU2 job proc_4d814817-46d4-4e86-bafd-7837330bb64a, started2026-10-24T00:49:59.626710924Z, deadline2026-10-24T00:59:59.626710+00:00 +TERM30. Runner elapsed282.153s. Source plan7084fd6c… was backed up at ee27624fe3974f767c5104203d6ddd6d9f68a835 before launch. GPU release is recorded separately.

F-coref contributes90,526,466 unique parameters. Combined with the frozen current production parameter inventory it would total9,687,994,456, excluding a prospective NCR addition. The old SFT32 adapter is a training-side control, not an extra production adapter added on top of full64. A unique coreference anchor remains a candidate identity; numerical/coordinate validity is not clinical attribution proof.

One initial evaluation guard failed before assembly, gold access or scoring because the old SFT32 checkpoint lacks the newer weighted checkpoint field partial_not_for_inference. Its original source and exact tool-returned traceback, plus the observed absence of pred/evaluation-started/metrics, remain in evaluation-attempt1-state.json and evaluation-attempt1.stderr.log. The separate old-schema entry checks completed_epochs=2/global_step=32/smoke_only=false; repair plan634466f… was backed up at78c33818946d3dd18c4f93beb89b1773df8555ae. Original run-plan and raw freeze stayed unchanged. This is one completed quality evaluation after one failed pre-score guard, with no NN repeat. See evaluation-control-binding.json for all exact SHAs.

See summary.json, relation-change-diagnostic.json, paired-predictions-freeze.json, responses-freeze.json, validation.json, metrics.json and prompt preparation/token sidecars. Large prompts, native outputs, predictions and raw responses stay external with exact SHA references; no further arm was run.
