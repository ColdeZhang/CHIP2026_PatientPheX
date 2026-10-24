# Fixed fulltext reverse 80-row inference

This source-only handoff reuses the successful new reverse probe at original index 19 and generates the remaining 79 rows using the fixed old SFT32 adapter `4f4815b6`. It does not reuse the normal fulltext probe. The disabled template becomes executable only after `bind_probe.py` verifies the complete recovered reverse probe and binds its eight immutable references. Parent owns binding, pushing and the one GPU start.

The original normal runner `9bb2b322` line 89 sorts by **ascending input token length, then row_id**. This source preserves that actual policy; the earlier verbal descending description was corrected during preparation before execution. Maximum input is 14390 with cap 14848, greedy generation 1600, native Qwen3.5 BF16/SDPA, matmul TF32 off and cuDNN TF32 on. Original normal used A800 and new reverse uses A30; hardware differences remain disclosed.

The absolute 1200-second deadline includes asset verification and loading, with soft stop at 1170 and TERM grace 30. Generated IDs precede raw response persistence, which precedes the original strict CID/PID parser. Invalid responses abstain for the whole row and continue; a wall stop or engineering error preserves partial evidence and stops without retry. Cached index 19 IDs/raw/format files are copied byte-for-byte, with new-run provenance recorded separately.

`actual/canonical.jsonl` retains the original normal six-field schema (`row_id`, `pmc_id`, `candidate_patients`, `strict_complete`, `abstention`, `provenance`). `responses.jsonl` retains original response fields and adds native decoding/input index/wall fields. Both follow processing order and have deterministic gzip companions. `record-manifest.json` binds all 80 per-row IDs/raw/format/canonical files by original index and row_id. The new terminal status is `COMPLETE_FIXED80_SFT32_FULLTEXT_REVERSE_NO_QUALITY`; expected coverage is 79 fresh plus 1 cached. No R1 codec or quality function is used.

Only local syntax compilation and static source/copy checks were performed while preparing this directory; no binder, tokenizer, model, remote job or quality evaluation was run by this helper.
