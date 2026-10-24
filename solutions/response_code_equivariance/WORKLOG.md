# 工作台账

更新时间：2026-10-24 UTC；分支 `explore/response-code-equivariance`，任务给定起点 `main@0878cbdf`。

## 当前状态

- **固定 cyclic1 负结案**：主 `B4 ∪ (原C∩cyclic1C)` Score **.701282498644**，低于B4 **.701325377212**；诊断cyclic1单臂 **.621756506260**。主micro +.000242887771、macro −.000414402044；新增关系 2 TP / 3 FP。两臂都未过门，不迁F0/dev/官网，不扫其他排列/seed。
- 唯一 .100 A30 GPU0 作业 `proc_d8c112f4-496a-43cd-874b-7c738d093e7d` / PID2743313 于13:36:35启动，13:43:57实际rc0，**442.208813秒**。53fresh+27原字节cache=80严格完整/0弃答；原1200+30预算内结束，GPU已释放。
- 模型仍原C32，全部8,997,081,600参数<10B；实际14848/1600。allocated22,283,378,176 B / reserved25,029,509,120 B，A30 reserved余量很小，不能只用allocated推断后续可行性。
- 253件终点/240shards、81缓存原字节全SHA；zip `3adf1ecc`/94,139B。双预测共同freeze `251dfb1d` 后新2v2/旧0，1189实体与非关系字节全保。当前完整证据位于仓库外 `research/response-code-equivariance-v1/gpu-v1`。
- 此为映射＋host/runtime开发控制：原C A100/Torch2.6，本次 A30/Torch2.7。只关闭本次固定配置，不把负结果归因为纯映射，也不否定已有原响应编码模型。
- 源/运行/工具检查点 `a1ee4f88` 已推送核验；完整质量与单一终审入口已落盘：`review-input.json` 4fd920ee；69件内容包 e777a0e3/745871B 已本地冻结。13:47:07备份offset622592写回UNKNOWN，正按原namespace库存SHA恢复；服务器ACK尚未取得，不称远端备份完成。

## 下一步

1. 此固定实验不再启动模型、parser、质量或其他排列。
2. 以仓库外 `review-input.json` 做一次增量终审；旧C和native证明沿用，勿重复运行。
3. 完成唯一内容包远端SHA/fsync/reopen回执，提交推送最终摘要；后继研究由root结合其他路线另定。

详见 [实验](experiments/E-20261024-EQUIVARIANCE-001.md) 与 [研究笔记](NOTES.md)。

## 2026-10-24 13:30 UTC：实际完整native CPU通过

唯一.100proc156e646f/PID2742802，120+5，实际11.7197s rc0/native10.9888s，8件全SHA。53新/27原cache、685372total/max14570/guard14848，原证据/IDs及逆映射完整。实际runtime2.7与原A1002.6差异明确。见CPU-REPORT.md及evidence/native-CPU-result.json；GPU/quality未启动，等待root排卡，原full64优先。

## 2026-10-24：获准唯一 GPU / 两固定质量，源已冻结

- 原 actual native 完整80已过；新增本地0.142047秒缓存/严格逆映射/生成AST合同通过，见 evidence/GPU-source-CPU-contract.json。没有重tokenizer、模型或质量。
- 固定 .100 GPU0 UUID `GPU-126293b9-6677-8464-cee7-01e693efbf29`，首次实际只读24164MiB空闲；启动器再核fresh无进程，GPU1不使用。原C32 adapter `2c777e88`，53 fresh/27历史三类文件原字节cache，14848/1600，唯一1200+TERM30。
- 入口见 evidence/GPU-ready.json，外部 `research/response-code-equivariance-v1/gpu-v1`；plan `a20c972a`，quality plan `e140395e`。本段为事前准备，尚无GPU job/新质量。
- 两完整端点共冻后主变化1v2（若B4字节exact则继承0）、诊断1v2、旧0。仍为mapping+host/runtime开发实验，无自动F0/dev/官网；诊断单臂若独胜B4也上报root。

## 13:36:35 UTC：唯一模型作业已启动

`proc_d8c112f4-496a-43cd-874b-7c738d093e7d` / PID2743313，91部署成员逐SHA/fsync/compile完成；源检查点 `876b02b8` 已推送且ls-remote核验。模型终点尚未到，检查原namespace见 `evidence/GPU-job.json`，禁止unknownstart重复。额外合成固定交集/非关系字节合同0.000642秒通过，无真实gold/评测。
