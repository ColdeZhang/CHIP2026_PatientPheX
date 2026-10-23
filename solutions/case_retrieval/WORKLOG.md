# 病例检索工作台账

实验 `E-20261023-008`；本轮从合并后的 main 建立独立分支。

- **目标**：固定 train64/dev16 与公共近似评测 v2 下，Score ≥ 0.8；保持训练/验证文献隔离。
- **状态**：进行中；固定基点 `00f7da8`，CPU四折选型已独立复现，Qwen3-8B全部源SHA通过；16篇内部fold0全文Qwen关系负结果已回收；同实体下.615159→.569358，过度归属，未采用。首日本地 0.613517597，官方 0.599432522，两者不可混比。
- **记录入口**：[实验过程](experiments/E-20261023-008.md)、[研究笔记](NOTES.md)、[小型证据](evidence/round2/README.md)。
- **运行位置**：外部 `/workspace/patientphex-runs/round2-20261023/case_retrieval/`，完整日志、预测、模型不得入 Git。
- **当前结果**：内部fold0原版 .613151 → compound修复 .615159；实体不变、关联微升，5项行为测试通过。固定 dev16 已得到 .625073738（原 .613517597），仍未达标；独立复评、重新拟合逐字节复现、36组内部输出复评全部通过。
- **下一步**：SFT与同构HPO完整控制已完成，直接并集负结果冻结；固定Sap .65/.92仅保留18新跨度，内部fold0固定规则关系Score .714060535（原.711353522），本路线同k21/.35完整链 .662396726（原.657638543）。待独立CPU复核；先完成整患者集合生成SFT可行性预算，再仅做一个固定fold1自身train48/原base初始化的同2epoch复验，不扫阈值、不提前评dev。
- **边界**：只修改本方案目录；共享变更与主线程协调；本轮不得上传官方。关键检查点及时 commit/push，回报版本与证据。

## 运行与恢复

- Qwen来源：10.5.2.100，mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，`case-retrieval/round2/`；恢复进程已完成，13文件全部源SHA一致，证据 `evidence/round2/qwen-download-complete.json`。
- 本路线仅用GPU1 `GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539`；GPU0现有他人进程，禁止使用；A800公共权重与运行时已由其它路线验证，当前分配其GPU2给本路线后续HPO标准化，启动前仍需核空。
- 解释器只读 `/home/zhangyuheng/projects/ScratchVL/.venv/bin/python`，基础模型的`PYTHONPATH` 指向本项目 `case-retrieval/round2/runtime`；adapter/训练需`case-retrieval/round2/training-runtime:case-retrieval/round2/runtime`两个完整绝对路径（见外部inference-plan.json的env），`PYTHONDONTWRITEBYTECODE=1`。
- 已完成对照：`llm-relations-v2-fold0/` 和 `llm-extract-v2-fold0/`（16篇90窗口，原始结果冻结；一次格式重试另存retry）。全文关系及新跨度负结果见实验报告。
- SapBERT FP16只读归一化模型从首日产物复制到同机自有 `models/SapBERT/`，全部文件已校验；内部fold0的43,437条HPO/训练别名CLS索引已预计算（6.85秒），已有1218个可定位抽取，391精确别名，827需语义候选/链接。

- 90窗口抽取及一次格式重试均完成，GPU训练开始前重新确认GPU1空闲；正式训练/续训进程和检查点将记录在下方。
- .3 LoRA smoke与重载已结束并将GPU0交回本体，产物`case-retrieval/round2-lora/smoke-bf16/`；小证据`lora-bf16-smoke-*`，不得当成正式模型或验证成绩。

- 13:31 UTC：真实epoch恢复小测试在 .100 `resume-test-bf16/`；2个训练原样本、epochs2/accum1/warmup0，pause1已原子保存step2，resume作业`proc_8362033d-6f57-49fa-a6d9-15d64f2cad71`。源码7c693a2、train_lora SHA2a25b83d...；真实resume状态审查通过，后续正式训练使用同入口；连续数值轨迹等价性未证明。

- 2026-10-23T13:38:11.752256666Z：正式305窗2epoch BF16启动 .100 GPU1，作业`proc_6e1059ff-40f5-47db-bfbd-c5363d13a35c`；远端`case-retrieval/round2/sft-extract-fold0-bf16-v1/`，每epoch保留adapter+optimizer/scheduler/RNG；大产物外存同名目录。启动前无他人GPU1进程。
- HPO基础模型链接69批已在 .3 GPU0独立运行，作业`proc_6f907c9a-4fe7-468d-8a28-15a5e4e321ce`，远端`case-retrieval/round2-normalization/full/`；先最短5729/最长8593tokens smoke全部合法无截断，保持同一固定prompt。

- 13:47:55 UTC：正式SFT epoch002固定90窗推断启动，.100 GPU1作业`proc_9329ac93-4399-46d1-8616-1534c5e855d3`，远端`sft-extract-fold0-bf16-v1/inference/`，原prompt SHA55bf2795...；仅推断epoch002，epoch001保留而未选择。

- 14:01:49 UTC（现已完成）：.100 GPU1唯一17窗格式恢复`proc_b672ed53-ccb7-4ed7-a2ad-9ef6e5694d6a`，远端`sft-extract-fold0-bf16-v1/inference/retry/`。原90响应已完成且永久冻结；外部inference-plan.json标注完成状态，inference-job.json保存实际原作业。

- 14:14 UTC：SFT后HPO的43请求已冻结（SHA61df3dab…，4155–7832tokens），A800 GPU2已从sequence正式接回；先最短/最长2条smoke，再batch1/maxnew1600同构正式链接。完整命令/env/基座与来源差异见外部`sft-extract-fold0-bf16-v1/hpo-inference-plan.json`；实际作业在同目录`hpo-smoke-job.json` / `hpo-inference-job.json`；14:16:27 UTC正式作业`proc_855aacdd-70a5-45bb-b204-80b6d205a02f`，2/2 smoke合法无触顶。

- HPO43批现已结束并回收完整SHA，A800 GPU2复查无我进程后交回root；.100 GPU1已交sequence共指试验。本方案当前无GPU作业。完整最终结果入口为r2-10与`sft-linked-filtered-control-summary.json`，恢复可直接按外部`linked-original/commands.json`和`sap-filter-control/repaired101/merge-evaluation-commands.json`重装。
