# 独立训练种子：CPU 准备

**查重未发现等价实跑；固定新seed已获准并于15:57:39在A100开始fresh训练，尚无新质量。** 初始CPU准备及其禁用GPU提案保留；当前实际入口见WORKLOG和evidence/formal-running.json。 同一个 Grounded32 `d52` 的随机采样、epoch 权重平均，以及不同数据的 full64/quote-completion，均不等于同配方 fresh 训练第二个种子。

唯一新训练 seed 固定为 `20261025`，原值 `20261023`。复用原 F2+F3 的 32 文献、87 患者行及全部既有编码/目标；标准 indexed CE、两轮、22 次更新、warmup2、r16、LR1e-4 均不变。生成 trainer 只有三处 seed 常量变化；[差异](evidence/seed-only.diff)和 [CPU/实际来源/成本](evidence/readiness.json)可直接核查。新权重及初始化张量尚未产生，不声称 CUDA 逐位可复现。

固定唯一质量公式：`B4 ∪ [G_new ∩ (G_old ∪ S ∪ Q ∪ C)]`。每折完整预测先冻结，再最多一次新 v2、旧0；若与 B4 字节相同则继承指标。主门是两折分别严格胜 B4（F1 .701325377212172、F0 .7281956289113009），macro 完整披露。F1 不正则不推进 F0；没有 G-new-alone、第三臂或 seed 扫描。供给集合在 B4 外为 133/141 条，**不是 TP 估计**。旧 G-only 交集仅为已替代、从未运行的准备提案。

单一恢复入口是 [WORKLOG.md](WORKLOG.md)，固定提案 [proposal.json](proposal.json)，具体协议和预算见 [实验记录](experiments/E-20261024-TRAIN-SEED-001.md)。原dev最佳仍为 .769235669942，本准备不产生新分数。

复建本地准备（使用一个不存在的仓库外目录；不启动模型）：

```bash
python solutions/training_seed_ensemble/prepare.py \
  --output /workspace/patientphex-runs/round2-20261023/research/training-seed-ensemble-readiness-20261024/rebuild-UNIQUE
```

入口校验原 trainer/编码/支持预测的 SHA，生成新 seed trainer、完整分组和禁用 GPU 的提案。已完成结果在外存 `prepared-v2/`，无须为接手重跑。将来执行还需独立 namespace、实际资产/资源绑定和有启动有效期的控制器；这里没有可误启动旧作业的入口。

评分调用计数更正：已完成F1为1主端点、1次`evaluate_report`、5次底层`evaluate`（1主+4固定sensitivity），旧基线0次；未改预测/主分/选择。F0新入口直接一次`evaluate(all/text)`，没有自动sensitivity。原件和更正见 `evidence/F1-evaluation-count-addendum.json`。
