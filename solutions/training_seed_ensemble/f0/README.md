# 固定新训练种子：F0 完整复验

F1 唯一固定端点已从 .701325377212 提高到 .704663449915，6TP/2FP；本目录只切换为完整 F0 16篇/47患者。仍使用新 seed20261025 的最终 `c3dc7de5` adapter，原两轮/22更新/双完整持久 ACK，不再训练、不选择 epoch。输入及原生 IDs 直接复用既有 F0 完整缓存：289420 tokens、最大9935；无重新分词、截断、子集或答案补写。`cpu-contract.json` 是纯元数据实际核验，原 native 证据保留，不冒称再次执行 tokenizer。

唯一端点仍为 `B4 ∪ [G_new ∩ (G_old ∪ S ∪ Q ∪ C)]`，F0 对照 .7281956289113009。原947实体/全部非关系字节保持，完整预测冻结后仅1次 `evaluate(all/text)` 主评分、0次report/0次sensitivity、旧0；完全等同则0v2继承。没有新单臂质量或支持集合扫描。原G解析器保留：cap、非EOS、整体JSON错整行弃答；合法数组中的不合法quote/重复anchor只拒绝该项。所有原始 IDs/响应先保存。

A100 `.12 GPU2` 待 SCST 实际终态释放，再核 fresh UUID/进程/≥2GiB磁盘。选择同A100是资源排期，并非已证明 A30 不适用。F1实测峰allocated21,014,939,136B，reserved未记录；F0最大输入小于F1的10080，cap仍1600。root允许真空闲A30时另绑定host，但本目录默认及源码仍是A100原Torch2.6.0+cu124/TF5.20.2，不偷偷换卡。

F0 固定2100+TERM30、启动TTL300秒，绑定时建立唯一绝对时钟。F1实际1319.293秒，按患者数代理1589.917秒；按输入token比例再加120秒代理1995.502秒，均不是时延保证。没有新held probe。卡将可用时再执行 binder，避免排队消耗TTL。

在本 worktree 中，令 `S=solutions/training_seed_ensemble/f0`、`B=/workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-v1`。唯一执行链：

1. `python "$S/bind.py"`；原双ACK、F1正门和 source-manifest 必须全部成立，写独占 `infer-plan.json`。`python "$S/remote.py" pack --plan "$S/infer-plan.json" --directory "$B/infer-F0-v1"`。推送源/实际plan后 deploy/start 各沿原意图。任何 start-intent 后禁止重start，仅 observe 恢复。
2. `python "$S/remote.py" deploy --directory "$B/infer-F0-v1"`，随后唯一 `start`；实际 receipt 到达后用当前 `sys.executable`、独立session与 `watch_inference.py --directory ...` 建唯一只读watch（owner.lock、180秒节奏）。不能重启已完成F1watch。
3. 终态先交准确 .12GPU2 release 给协调者，再 `python "$S/collect.py" --directory "$B/infer-F0-v1"` 完整收件。不是只见成功rc就做质量。
4. `python "$S/postprocess.py" extract-bind`；设 `G=/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation`，`PYTHONPATH=/workspace/patientphex-worktrees/explore-grounded-generation python "$G/f0/extract_quotes.py" --plan "$S/extraction-plan.json"`。
5. `python "$G/f0/held_prepare.py" --extraction-manifest "$B/F0-postprocess-v1/quotes-v1/manifest.json" --output-dir "$B/F0-postprocess-v1/normalization-preparation"`。仅完整必要queries进入原public41099 Sap，固定b59政策，不读取gold。
6. `python "$S/sap/bind.py"`；若非零queries，先协调 `.100 GPU0` 实际释放，再原180+30/TTL300的 pack、push、deploy、start、完整collect。若零queries直接后继，不启动空GPU任务。该host/权重/FP32 CLS_L2/max32沿实际成功F1 206条；版本差异已在原host合同明确。
7. `python "$S/postprocess.py" normalize-bind`，`python "$G/f0/held_finalize.py" --plan "$S/normalization-finalize-plan.json"`；`postprocess.py assemble` 完整冻结，然后唯一 `postprocess.py score`。不存在任何第三臂。

当前源冻结入口 `source-manifest-v2.json`（旧v1与CPU历史保留）；CPU证据 `cpu-contract.json`、`evidence/endpoint-contract.json` 和新增 `evidence/call-count-contract.json`；父F1稳定结果 `../evidence/F1-result.json`，实际122件备份 `../evidence/F1-backup.json`。主要产物外存。此准备阶段尚无F0推断、归一化或评分。

F1历史“1v2”计数应解释为1主端点/1次report/5次底层评分；只做文档更正，无重跑。F0版本2同时纠正准备源码中误将指标键折名化的menF0/docF0，真实指标始终是menF1/docF1；输入、训练、推断和固定端点均未变化。
