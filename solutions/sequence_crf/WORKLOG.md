# CRF 序列标注工作台账

实验 `E-20261023-009`，分支 `round2/sequence-crf`，基点 `00f7da8`。

- **目标/边界**：原 train64/dev16、HPO 2026-06-23、evaluator v2 all/text，Score ≥ .8。训练内四折用于选择，禁止输入金标进入推理；本轮不上传官方，未达标不申请合并。
- **当前结果**：最高 dev **.700635947**（CRF + 同折 SapBERT 新跨度 + 复合标签修复 LR），冻结源 `654e93d`；独立验证已重训基础CRF、重装增强输出，字节SHA完全复现。内四折规则优于LR，dev相反；明确记录dev参与候选选择。
- **后续进展**：上下文患者LR四折较原LR全提升，选择加入NO负例变体，平均 .659744，一次dev复验 .687185，低于当前最好，未按dev回扫。五标签 NER .580778、HPO5000预适应 .572053，均为 fold0，对比CRF无优势，保留负结果。
- **当前待办**：BioLinkBERT-large 通用底座8epoch fold0 .557760，整体负结果；HPO重排对CRF池 .637749负、对弱NER池 .562669仅微增；三NER新跨度≥2票补充仍负，Qwen58条已完成 .618120负结果，GPU已释放；关系集合控制无增益。29条扩展不盲跑，12条thinking配对已完成，仅37/85候选可答、6提示触顶，时间增12倍；共同可答子集FP减4/TP减1，不推广。官方O编号锚点四折两升两降，不升级默认，不跑dev；表格Task2补充fold0两关系版本均下降，不推广。正准备严格嵌套OOF候选的有监督Qwen关系LoRA。新诊断显示已有正确跨度768/778规范化正确，重点是跨度漏检/误检。未使用赛事80篇训练的公开成品权重。
- **基线注意**：baseline-fold0 是未拆复合关系标签的旧版 .613929，fold1–3已修复；已补跑修复版fold0 .603398，实体不变、关系变化，比较时需注明。
- **复现/证据**：[实验过程](experiments/E-20261023-009.md)、[笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。完整预测、模型、过程日志位于 `/workspace/patientphex-runs/round2-20261023/sequence_crf/`，不入Git。
- **接手顺序**：先读根 HANDOFF.md/AGENTS.md，再读本页和E-009当前结论；最高候选为 `crf-semantic-dev/lr.pred.jsonl`，独立审计在根外存 `validation/crf-semantic-review.json`。GPU作业/源码/缓存manifest按各实验目录查看，不可复用无指纹的旧模型。
