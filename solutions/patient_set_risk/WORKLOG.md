# 工作台账：患者集合风险优化

状态：CPU/公开资料准备完成，单4患者8序列供给与梯度probe提案已定，**未启动GPU或训练**。主分支基线21dca5f3，独立branch `research/patient-set-risk`、独立worktree；教师供给负结果留 `research/verified-rationale@d20e7634`。

目标：针对关系分项，判断完整患者概念集合F1的策略梯度是否提供旧CE/推断expected-F1/DPO之外的新监督。成功标准先是可区分、同义一致、实际采样可用的训练reward与正确梯度链，不拿训练成绩当验证成绩。

- 已读当前main AGENTS/HANDOFF/WORKLOG/MAP，核旧expected-F1仅推断cut，旧DPO固定同cardinality交换。
- 选择G87完整患者的SCST，只保一个方案；暂停BioModernBERT等另一架构调研。
- 原train32纯CPU：818完整v2 gold原子、5空患者，原target归一化mean F1 .9946025是监督供给自检；82删除/71外患者新增可改变reward，32不同quote同概念不改变reward。0模型/0fit/0heldgold/0quality。
- 初次Python/NumPy ABI导入失败在读数据前；固定系统Python3.13成功。最小四患者选择在任何GPU前按shape轴改成76/44/83/19，原记录保留。
- 来源及数学见 [REPORT](REPORT.md)，唯一参数/预算见 [pilot-plan](pilot-plan.json)。提出1800+30共享上限，非实际已启动deadline。
- 当前等待root审阅具体提案；即使准备通过也不启动87全量、174生成或held质量。A800 GPU2当前归sequence F0-47，必须实际交接后才可fresh门；新quote正常化需要原公共Sap完整查询。

实际CPU命令：

```bash
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 solutions/patient_set_risk/census.py --grounded-source /workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation --run-root /workspace/patientphex-runs/round2-20261023 --hpo /workspace/patientphex-data/hp.obo --output /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-readiness-v1/cpu-v3
```

现目录已存在，恢复读取现有结果，不为查重重算。全部训练案例125KB、公开页面/代码在仓外；Git只放源码、固定计划、短证据和记录。
