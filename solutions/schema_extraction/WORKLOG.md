# Schema extraction 工作台账

更新时间：2026-10-24 UTC。分支`explore/schema-extraction`，仅维护`solutions/schema_extraction/`。原dev全局最佳仍以root台账为准，本线未访问原dev/官网。

## 当前状态

- **E001完整负结案**：固定GLiNER2.5提取＋train32/publicSap规范化，F1完整16/39/164窗；N∪new Score **.672407144**低N .676546355，new-alone关系 .459793527。两端点共冻后各1v2、旧0、原dev0、官方0。新增47=12TP35FP；不是本线涨分。
- **完整证据已回收和独审**：496件逐SHA、全部native候选/坐标/原分数；GLiNER79.625秒、Sap7.705秒，GPU均释放。独审dac8bde8通过，见[终态指针](evidence/terminal.json)、[质量摘要](evidence/quality-summary.json)。原managed exit_code未暴露，running=false及release另有实证。
- **E002完整F0供给已冻**：原固定方法16/47/172窗，GLiNER68.948秒＋40-query Sap8.904秒，规范化75关系。new-alone `3c792094`固定强F0壳94504b97的947实体，格式零错零警；全部520件原候选与坐标证据逐SHA回收，两个GPU释放/原proc running=false。本线F0评分0、未读gold；root固定共识F0 .711507580→.710063636，新9=3TP6FP，门失败关闭，不迁原dev/不改cut。见[完整终态](evidence/f0-terminal.json)。
- **Task1监督准备CPU完成**：320窗/2642目标、max49 gold、max698词片、74空正窗；完整原生目标合同8.353秒、loader320×10全覆盖0.038秒。38词内正标注无法精确表达并保留审计；0模型加载/forward/fit/质量。见[准备指针](evidence/task1-readiness.json)。后继已由root分配独立gliner25_task1工作线，本线旧终点保持。
- **最近已核远端检查点**：7eb4039，CPU准备收尾待本次阶段push；历史实际运行源179bd60f/协议dd526a45不可变。

## 恢复入口

- [E001方法/过程/负结果](experiments/E-20261024-SCHEMA-001.md)、[E002固定F0计划](experiments/E-20261024-SCHEMA-002.md)、[NOTES](NOTES.md)、[f0-plan](evidence/f0-plan.json)。
- 外部根`/workspace/patientphex-runs/round2-20261023/stage8-schema-extraction`。E001已完整`full100-all/`496件、`normalization-result-v1/`、`quality-v1/`、`terminal.json`。不要重跑已完成E001或重下已核资产。
- .100 mount5a9955da-5240-41cc-9e52-05d3d8da8a3c，自有PROJECT/stage8-schema-extraction/prepare-v1模型/runtime复用；F0已完成独立目录f0-v1。原proc_7e60980f已结束。GPU1 UUID fc5d0525，F0结束立即归还Qwenowner。
- .3 mount1593f159-19bc-41ca-8b4c-2e7455f5db99，自有normalization-v1含已核train32/normalizer，F0独立产物。公共bank41099与Sap模型复用，绝不重编码。
- UNKNOWN先恢复原意图/进程/已存chunks，不重复start。旧.3传输服务TTL结束，临时token已全部清理。

下一步：本线E001/E002固定配方关闭，root消费结果见[evidence/root-f0-consumption.json](evidence/root-f0-consumption.json)。CPU准备已交独立explore/gliner25-task1后继；本线无新GPU/质量任务，不重跑旧源或修改冻结预测。
