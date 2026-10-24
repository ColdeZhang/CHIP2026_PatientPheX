# Frozen train32 + F1 native coreference input census

Completed all 48 documents once in 10.154s on fresh-empty A30 GPU1, inside the 600s cap. The separate first-document CPU probe completed in 13.013s. This report measures unlabeled input coverage only; it does not estimate patient attribution or task quality.

| Inputs / mapping | Patients with additions | Docs with additions | New nonoverlapping mentions | Unique-patient clusters | Conflicting clusters |
|---|---:|---:|---:|---:|---:|
| train32 / A | 28 | 19 | 233 | 33 | 3 |
| train32 / B | 51 | 30 | 578 | 79 | 9 |
| F1 / A | 11 | 8 | 173 | 14 | 0 |
| F1 / B | 26 | 15 | 392 | 37 | 5 |

All 417 original mention records for 126 supplied patient IDs were preserved. Every derived candidate input was reconstructed and reversed locally from its sidecar with exact original input equality. All native logits were finite. Rejected coordinate errors: `{'span crosses passage/separator or has no mapping': 18}`.

A uses exact original anchor intervals. B uses full containment of given anchors and unions patient IDs over the entire cluster; multiple patients/shared coordinates reject the entire cluster. Both reject cross-passage/separator spans and additions overlapping old anchors or each other. Literal aliases never establish identity. The NP used for B containment itself overlaps an old anchor and is not added. Synthetic A=0 remains preserved; B passed known-multiple-patient NP/shared-coordinate/cross-passage rejection checks.

Addition-set comparison: `{'train32': {'A_only': 8, 'B_only': 353, 'both': 225}, 'all48': {'A_only': 38, 'B_only': 602, 'both': 368}, 'F1': {'A_only': 30, 'B_only': 249, 'both': 143}}`. Added mention character lengths and native length distribution are recorded in SUMMARY.json. These are candidate spans, including ordinary referring expressions; no semantic correctness labels were read or inferred.

Actual F-coref parameters 90,526,466; current production plus F-coref = 9,687,994,456. This excludes a prospective NCR addition. Model revision e91dfff12879495d882ba9460d0c5d5dd44ade59 and native source ae224139196d122b225af5a7e73b5fc0b6e1076d stayed fixed; BF16/singleton/maxsegment512/maxdocNone/maxbatch10000, no truncation. GPU allocated peak 1,402,296,320 bytes. CPU/GPU numerical kernels differ, so the CPU probe is timing evidence only.

No original-dev data, train/F1 gold entity/association labels, 9B prompt generation, LLM training/inference or quality scoring was used. This output supplies reversible identity candidates only. A unique known anchor cannot rule out unanchored other people in a cluster or establish clinical ownership. Any downstream prompt use requires separately frozen provenance and coverage inspection; task quality remains unknown.

Detailed native clusters/mapping sidecars remain in external `full48/documents.jsonl` and compressed copy; their SHA/bytes and all source/input references are in SUMMARY.json. The original CPU plan/probe, GPU device-only patch, exact owner backup commit, fresh resource check, actual job/deadline, logs and terminal status remain separately available.
