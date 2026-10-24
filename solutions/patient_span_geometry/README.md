# 患者跨度全锚点几何残差

目标是改善患者归属，主成功门为固定 F1 强 B 的 Score `.686338981272`。32篇 F2/F3 训练、16篇 F1/39患者完整留出。这个内部折是开发验证，不是原 dev 的0.8验收。

原24轮患者条件跨度实际源码复制于 `evidence/reused-actual-source.json`，原模型逐字节保存为 `vendor/base_model.py`。唯一新机制为50参数距离桶×同原段残差，全部唯一患者锚点平均；原参数3e-5，新增表固定1e-2。这是含独立学习率组的组合机制，不称相同优化器的纯特征消融。全部原图、max-MIL、全局GP损失、24轮768步76warmup、双logit>0及规范化不变。

执行入口：`contracts.py --encoded EXISTING --output REPORT`（CPU无模型权重）；`driver.py --mode formal --plan plan.json --source SOURCE --encoded EXISTING --output NEW_RUN`（唯一900+TERM30）。复用原两个真实最大图压力检查，不新探针、不重分词。`assemble.py`冻结两完整预测后由`quality.py`唯一各1v2；旧B/8轮/24轮只读。

外部根：`/workspace/patientphex-runs/round2-20261023/research/patient-span-geometry-v1`。具体源、资产、数据隔离与预算见`plan.json`。完整raw、checkpoint和日志均不进Git。

本次已完整负结案：主.676794409655 < 强B.686338981272；诊断.507641213990。见evidence/quality-summary.json与实验记录。
