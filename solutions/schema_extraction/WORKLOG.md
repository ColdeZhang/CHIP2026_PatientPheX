# Schema extraction 工作台账

更新时间：2026-10-24 UTC。分支`explore/schema-extraction`，仅维护`solutions/schema_extraction/`。原dev全局最佳仍以root台账为准，本线未访问原dev/官网。

## 当前状态

- **E001完整负结案**：固定GLiNER2.5提取＋train32/publicSap规范化，F1完整16/39/164窗；N∪new Score **.672407144**低N .676546355，new-alone关系 .459793527。两端点共冻后各1v2、旧0、原dev0、官方0。新增47=12TP35FP；不是本线涨分。
- **完整证据已回收和独审**：496件逐SHA、全部native候选/坐标/原分数；GLiNER79.625秒、Sap7.705秒，GPU均释放。独审dac8bde8通过，见[终态指针](evidence/terminal.json)、[质量摘要](evidence/quality-summary.json)。原managed exit_code未暴露，running=false及release另有实证。
- **E002进行中，固定F0复验**：root新共识融合F1 .686338981→.690618522，故优先提供F0关系；不改本线原负结论。F0纯输入16篇/47患者/172窗，train32无交集；沿原模型/全部策略，不训练/调阈值。root已排.100 GPU1完整600+30，随后.3必要Sap180+30；先CPU合同、源计划push、fresh门。质量由root统一，**本线F0评分0**。
- **监督API仅CPU可行性已完成，暂不扩展**：原生boundary训练入口可用；文档级关系不能冒作精确mention-pair标签。0模型加载/forward/fit。详见E001末节。
- **最近已核远端检查点**：68b3e850，当前收尾与F0数据合同待阶段提交；历史实际运行源179bd60f/协议dd526a45不可变。

## 恢复入口

- [E001方法/过程/负结果](experiments/E-20261024-SCHEMA-001.md)、[E002固定F0计划](experiments/E-20261024-SCHEMA-002.md)、[NOTES](NOTES.md)、[f0-plan](evidence/f0-plan.json)。
- 外部根`/workspace/patientphex-runs/round2-20261023/stage8-schema-extraction`。E001已完整`full100-all/`496件、`normalization-result-v1/`、`quality-v1/`、`terminal.json`。不要重跑已完成E001或重下已核资产。
- .100 mount5a9955da-5240-41cc-9e52-05d3d8da8a3c，自有PROJECT/stage8-schema-extraction/prepare-v1模型/runtime复用；F0将用独立目录。原proc_7e60980f已结束。GPU1 UUID fc5d0525，F0结束立即归还Qwenowner。
- .3 mount1593f159-19bc-41ca-8b4c-2e7455f5db99，自有normalization-v1含已核train32/normalizer，F0独立产物。公共bank41099与Sap模型复用，绝不重编码。
- UNKNOWN先恢复原意图/进程/已存chunks，不重复start。旧.3传输服务TTL结束，临时token已全部清理。
