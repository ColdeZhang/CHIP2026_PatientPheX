# BOUNDARY 完整删除控制

唯一公式在 [decision.json](decision.json) 预锁：从强 F1 的原 1313 个实体中，只删除同一 `(pmc_id, offset, length, text)` 的完整、严格 `BOUNDARY` 输出所对应的全部实体对象。BOUNDARY 表示此跨度未获支持，与临床 NO 独立。其他实体、关系、患者、文档对象及顺序均保持。没有新增实体、替换 ID、阈值或 HPO/患者子集。重复输入 row/span 或 raw/parsed 身份冲突直接失败。

执行只依赖 Python 标准库。`schema.py` 是父方法冻结 parser 的逐字节副本；其余代码独立，不修改父方法和旧 veto。已完成的来源核查与合成合同见 [台账](WORKLOG.md)。当前已完整执行F1并小幅正；见 [结果指针](evidence/result.json)。下列为已执行恢复入口，不要覆盖原排他intent重跑。

父 owner 完整回收 1416 行 `results.json`、`complete.json` 和全部 `generation/0000..1415.{raw,parsed}.json` 后，root 先核原训练完整终态、两轮 ACK 与收集身份，再执行：

```bash
S=/workspace/patientphex-worktrees/explore-boundary-veto/solutions/boundary_veto
D=/workspace/patientphex-runs/round2-20261023/research/mention-boundary-selection-v1
O=/workspace/patientphex-runs/round2-20261023/research/boundary-veto-v1
mkdir -p "$O"
python "$S/run.py" plan --inference "$D/gpu-v1-collected/infer" --output "$O/F1-v1" --plan-output "$O/plan-F1-v1.json"
python "$S/run.py" assemble --plan "$O/plan-F1-v1.json"
```

`plan` 只绑定原件 SHA，不解析响应选择政策；`assemble` 核完整原件、encoded row、输入 quote、原始 token 元数据和严格解析，装配并验证唯一完整预测，持久发布 `freeze.json`。两阶段均禁止打开计划中的 gold。每次新输出目录排他创建；失败保留原件，修复必须单独记录，不能覆盖重来。

只有 root 完成来源静核后才执行以下质量入口；此准备任务没有运行它：

```bash
python "$S/run.py" quality --plan "$O/plan-F1-v1.json"
```

有删除时仅一次冻结公共 v2 `evaluate(all,text,wildcard_no_id=True)`，不调用带敏感性多次评测的 `evaluate_report`，不重算旧分数；零删除逐字节复制旧预测并继承旧 metrics，零 gold 打开/零 v2。`quality-intent.json` 排他写入防止重评。父 strong-union/new-only 两终点保持独立，不能用于选本控制的子集。Score 严格高于 .7286975364690524 只是当前 F1 成功门；本控制不自动扩 F0、原 dev 或官方上传。

合成合同可重现到一个不存在的输出文件：

```bash
python "$S/contracts.py" --output /tmp/boundary-veto-contracts-new.json
```
