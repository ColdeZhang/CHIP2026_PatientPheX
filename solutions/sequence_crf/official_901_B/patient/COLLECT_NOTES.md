# Patient B 完整原件收件入口（仅源码准备）

`collect.py`只收原作业，不start/stop进程、不查GPU、不NN/重生成/codec/评分。当前仅本地AST静态检查，未导入执行collector、未调用MCP；实际运行由父线程在既有monitor确认完整终态后负责。原infer5b899、driver46abb、export caf48均未修改。

未来命令：`python collect.py --run-dir P --client-dir /workspace/patientphex-tools`；默认读取`P/infer-plan.json`，也支持`--plan`。`--inventory-only`只盘点，不创建远端包。默认read_file 128KiB，`--chunk-bytes`最多256KiB；不自动重试。

远端`patient/control/infer/**`完整映到本地`P/inference-collected/control/**`，远端`patient/prediction/**`完整映到`P/inference-collected/prediction/**`。包含全部420 raw/generation/decision与responses、indexes、manifest、complete、全部control日志/终态。先核rc0/无timeout/完整420、driver绑定原plan，并单次读取该原driver PID的`/proc`确认已退出或僵尸；仍能运行则返回RUNNING，不mkdir/zip、不轮询，避免wrapper写完后最后print造成日志变动。再核raw-first metadata及逐row对应；只在通过后创建唯一`patient/complete-collection/patient-complete.zip`及manifest。ZIP已有则只校验库存，绝不重打包；包装残留pending目录会停下，原件保留。UNKNOWN立即停；下次先inventory。明确LUNA记录≥180秒退避，不阻塞sleep。

本地原包、每段bytes/SHA收据和partial均持久；允许只续缺块/成员，不重新执行模型。ZIP安全路径/重复名/符号链接、所有成员size/SHA、fsync文件及目录祖先核验。最终`P/inference-collection.json`含export所需顶层`local`和`files`；其完整ref只能收件实际完成后填写，不能预造。原监控owner不变，本入口不并发轮询。

export唯一后继：保持现`export_positive_keys.py`不变，其plan需`output`独立目录（如`P/positive-export`）及以下13个`references`。`collection`指实际`P/inference-collection.json`，size/SHA待终点；其余12件当前已读真实字节核验如下。export会再核全420原始ID/decode/codec，后继空间证明另由父线程完成；collector不替代该合同。

路径缩写：`S=/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/official_901_B/patient`；`R=/workspace/patientphex-runs/round2-20261023`；`P=R/research/official-901-B/patient`；`F=R/research/patient-first-full64-memory-recovery/formal/training-terminal`。

| references键 | 实际位置 | bytes | SHA-256 |
|---|---|---:|---|
| inference_plan | `P/infer-plan.json` | 8324 | `1e4e67c2b315e564f7430b35965efab06471d1297db4f365de183f4eb1f73652` |
| runner | `S/infer.py` | 21397 | `5b899375fd82121ca8119d85aa5f62a3c0f3c725544a6bbb73516928f77946c7` |
| schema | `S/patient_first.py` | 7505 | `25802e682fcd801351377ce3a9f1815a2433c44e8d73c50ab35fba5016358f08` |
| codec | `S/limited_codec.py` | 4601 | `60731e1297930415afd8bb8c2a979814d7c8cb9a8c132c6c6debbd85d50cb289` |
| raw_store | `S/raw_store.py` | 1657 | `d52bb0c186d2467b5ae5710a6b7df2ee271783c22ceadadc2e56299295e3af43` |
| prompts | `P/prepared-collected/prepared/prompts.jsonl` | 5403066 | `9f94fffc474b1b3503032f492dfcdc6355f205c72abe8203d1c0f1aa72761d86` |
| token_audit | `P/prepared-collected/prepared/input-token-audit.json` | 121782 | `ccafddec3e9f66e1176becf982e1ab1aae4bbb9e209d8ece054e78b316c27932` |
| input_contract | `P/prepared-collected/prepared/input-contract.json` | 2115 | `eabbfa5fc713aecc2a36cd2a70ca2632b2693dc5a7d3df007fa4d9bb384ab106` |
| training_complete | `F/training_complete.json` | 527 | `9c9c7d356d373f10d81f1ad987eb04ac4b9ab4fb3d22c7240e9ae810383376a3` |
| training_checkpoint | `F/epoch-002/checkpoint.json` | 12697 | `0f8cb039da6db2552eb8ef6693e087c4dc53b6b083ecb535896d131fff7f954d` |
| training_durability | `F/durability-epoch-002.json` | 2234 | `bd174b1fd07bb77b5494c8f6ade1627ebf010b71de3a2a3f710e5e20ab5cd8be` |
| tokenizer | `R/case_retrieval/qwen35-preparation/verified-small/tokenizer.json` | 12807982 | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |

收藏入口本身只做静态核验，不声称已真实下载或校验远端1200余件。完整收据会保存实际文件数、原ZIP SHA与目录映射；任何首次错误保留证据后停，父线程据库存决定恢复。
