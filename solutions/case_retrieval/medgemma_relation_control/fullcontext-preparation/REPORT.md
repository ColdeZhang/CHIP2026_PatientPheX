# MedGemma strict32 append-only full-context CPU preparation

This is one input/tokenizer audit, separate from the ongoing original-input MedGemma training. It does not train, infer, score, truncate input, change the current trainer, or authorize a new training run. The native CPU job completed once on `.100` in **13.458 seconds**, with CUDA hidden and uninitialized and no model weights loaded. Job: `proc_ce41d827-6997-40c8-91a9-2308d4c81b3c`; remote hard timeout was 60 seconds with no grace or retry.

| Scope | Original tokens | Full-context tokens | Longest full-context sequence | Ratio |
|---|---:|---:|---:|---:|
| 124 training rows, one epoch, including target | 382,000 | 965,436 | 12,824 | 2.527319 |
| 80 F1 input-only prompts | 238,727 | 645,000 | 13,805 | 2.701831 |
| Training supervised suffixes | 7,150 | 7,150 | 109 | 1.000000 |

All 124 native prefixes, masks and supervised turn boundaries passed; each row retains its original supervised-token count. The appended training prefixes total 958,286 tokens. Training maximum is `12628480:relations:8`; F1 input maximum is `12455887:relations:40`.

The existing trainer rejects sequences above 8,192: **51 of 124** expanded training rows exceed it, including two above 12,288. Consequently the expanded data cannot be passed unchanged to the current training entry. No cap was changed. All 80 inference inputs remain below the existing 14,000 input cap. Adding the unchanged maximum generation allowance of 1,600 gives a longest possible input-plus-output length of 15,405. The model config's 131,072 position limit is not evidence of GPU memory fit or useful accuracy.

The original-input MedGemma first epoch measured **158.371390 seconds** for its training loop, sourced from `formal-preparation/formal/train-001/progress.json`. Multiplying that by the new token ratio gives only a **linear token proxy** of 400.255 seconds per epoch, or 800.510 seconds for two loops / 1,930,872 full tokens. This is not observed full-context training time: attention length and allocation may behave nonlinearly, and model loading, checkpoint saving, persistent export and inference are excluded. In particular, the separately observed 118.342-second checkpoint export must not be multiplied by the token ratio. No memory prediction is claimed.

The construction reuses the previously reviewed `prepare_full_context_control.py` source (`ae7f5853...`) and its exact append prefix, passage ordering, offsets and patient-marker overlap policy. It keeps every original CID, PID, candidate, answer string and original message prefix; removing only the appended suffix restores all 204 original rows exactly. The complete supplied `full_text` array is appended, without section selection or omission.

Input provenance:

- Original strict32 training124: SHA `cb7f67e1...`; original F1 prompts80: `ddba03ff...`.
- Existing pure train32 input: `sft-meta1-fusion/public-training-candidate-preparation/train32.input.jsonl`, SHA `a2c2baec...`; original F1 input: `fold-1/validation16.input.jsonl`, SHA `962b76a3...`. Both match their document subsets of the frozen pure train64 cache, and the 32/16 document sets are disjoint. No F1 gold was opened.
- Training: 87 patients, 1,122 passages, 255 given mentions. The frozen overlap rule inserts 254 markers and skips one overlapping span; every patient remains explicitly marked. F1: 39 patients, 570 passages and all 162 mentions marked. Every supplied mention is an exact original coordinate slice, and all passage text reconstructs exactly.
- Native interface: `aecef7ea...`; fixed model asset inventory: `expected-files.json`, SHA `690e0829...`. Only its **eight tokenizer/config files** were actually rehashed and loaded in this audit, not the model weight assets. The existing native Transformers 5.20.2 environment was reused with offline flags; no download or model class loading occurred.

Nine local construction contracts passed before the remote job. The native job independently reconstructed the appended rows and coordinates from the original rows and pure input, checked the eight actual asset SHA values, and used the unchanged native `encode` / `render_prefix` functions. The original tokenizer baseline audit was reused rather than rerun.

Evidence: `preparation.json`, `cpu-launch-plan.json`, `remote-source-verification.json`, `execution-intent.json`, `known-job.json`, `process-terminal.json`, `terminal-receipt.json`, and `actual-cpu/summary.json` (SHA `5d295ec580cd0c862a0fa1ded99ccfe93bb68f92631aec8dbbc5eb94868d34d9`). The 108,287-byte per-row token audit remains external at `actual-cpu/row-token-audit.json`, SHA `80f5bbce3f2f4d1537bfef9e9dbee5bd61c77d17bb7430c3810fc6bfbdedd1e0`. Full texts, compressed bundle, per-document audits, transport chunk receipts and ordinary logs stay outside Git. The remote namespace is `research/medgemma-relation-control/fullcontext-preparation-v1`; no HTTP server was opened and no other experiment's files were modified.
