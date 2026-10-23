# CRF 序列标注工作台账

实验 `E-20261023-009`，分支 `round2/sequence-crf`，初始基点 `00f7da8`。当前已同步共享主分支 `d13ab17`，合并检查点 `338256e` 已推送；方案代码不因同步而变化。新任务还应读取 `git show main:HANDOFF.MD` / `main:AGENTS.md`，再看本页。

## 目标与可信最佳结果

- 原 train64/dev16、指定 HPO 2026-06-23、公共 evaluator v2 all/text，**Score ≥ .8** 才申请主线程验收合并。本轮不上传官方。
- 最高 dev **.700635947**：CRF＋同折 SapBERT 新跨度＋复合关系标签修复 LR，源码冻结 `654e93d`，预测 `crf-semantic-dev/lr.pred.jsonl`。独立审查已从训练数据重训基础 CRF、重装增强输出，字节 SHA 完全复现。
- 内四折规则关系优于 LR，原 dev 相反；LR/rules 都评过原 dev，该选择使用过 dev，不能声称盲验证。尚未达到 .8。

## 正在执行与恢复入口

1. **共指控制**：.100 GPU1 正在处理 fold0 输入的 72 个文本窗口，进程 `proc_c31bce6c-97fe-4868-a51f-5bf3d5311cb3`。恢复入口：外存 `coreference-fold0/job.json` 和 [小型运行索引](evidence/round2/coreference-fold0.json)。输入、提示和源码已冻结；完成后整份回收核 SHA。
2. **CPU 控制**：只改患者 mentions，固定原 947 实体。比较原规则、生成 mentions 继续触发旧全文 alias 扩充、生成 mentions 仅真实坐标三路径；原给定 alias 处理保持不变。无新增 mentions 时三路已逐字节复现原规则预测。实际预测先冻结再评分。
3. **SFT 控制**：共指完成后复用现有 epoch2 adapter、原 58 候选提示协议，仅改变 mentions，不重训。A800 GPU2 优先检索路线集合 SFT，需明确交接和再次查进程才启动；不迁设备制造新的配对差异。

## 已完成的近期结论

- 关系 LoRA：204 条严格嵌套 OOF 提示独立审查通过，2epoch/52steps 已完成。同 A800 基座 `.623068` → adapter `.674333`，减少过度归属但低于同实体规则 `.711354`；固定集合投票也未超规则。原 dev 未新增测试。
- 本体校准池的 24 个新概念/26 支持跨度经现有 adapter 判断，保留原实体及规则关系，仅增 14 项关系（4TP/10FP），fold0 `.712942`。边际精度较低，尚不推广。
- 训练锚点诊断独立复算通过：967 个 SFT 候选内正关系中，363 项实际局部证据无目标患者 marker；全局 anchors 仍可见，不能称不可回答率。完整全文 LLM 共指此前没有实际运行。
- 共指首轮训练输入 pilot 因子串定位与身份混淆失败；同 3 篇仅修订一次为唯一原文 context＋完整词边界，严格保留 4 项新增且逐条身份检查通过，召回保守。现按此协议一次复验 fold0，不继续按结果调提示。
- 神经 NER、HPO 重排、零样本关系、thinking 小样本、O 编号锚点、外部表格等正负结果与限制均保留在实验报告。O 编号四折不稳定、表格新增关系均为本地 FP，未升级默认。未使用包含本地 dev 的赛事公开成品权重。

## 证据与下一步

- [实验过程](experiments/E-20261023-009.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。完整产物根目录：`/workspace/patientphex-runs/round2-20261023/sequence_crf/`；权重/预测/原始响应/完整日志不入 Git。
- [SFT 固定结果与恢复索引](evidence/round2/relation-sft-fold0-results.json)、[新关系池结果](evidence/round2/new-ontology-relation-results.json)、[训练锚点诊断](evidence/round2/patient-anchor-audit.json)。A800 现有 base 和本路线 epoch2 adapter 可只读复用；旧 180 概念/29 提示弱池未运行。
- 共指完成后保留全部控制，包括负结果；如无可靠收益，停止当前原型，不以格式通过替代身份或分数验证。后续再协调检索新跨度与本体候选组合，避免混淆本次单变量对照。
- 历史基线口径：旧 baseline-fold0 未拆复合关系标签 `.613929`，修复后 `.603398`，实体不变；fold1–3 已使用修复版。不要将四折旧关系基线写成同一未修补模型。
