# 固定三路关系并集：双折通过预定门槛，尚未迁 dev

唯一新组合是 literal union(oldSFT, same-adapter reverse, patient-first)。这是看到此前 reverse-union 和 patient-union 两个增量结果后提出的 posthoc 控制，非原初 prequality 注册；两折均已反复用于研究，不能称为全新盲测或泛化证明。

| 折 | old∪reverse Score | 三路 Score | ΔScore | Δmicro | Δmacro | 同池 rules Score |
|---|---:|---:|---:|---:|---:|---:|
| F1 | 0.662472447 | 0.663841821 | +0.001369374 | -0.007934237 | +0.013411732 | 0.644761594 |
| F0 | 0.704687440 | 0.709556743 | +0.004869303 | +0.005050889 | +0.014426323 | 0.711353522 |

预定 keep 是两折 Score 均严格超过各自 old∪reverse，且两折 macro 均不降；实际通过。F1 新增 35 个关系仅 8 TP/27 FP，micro 下降；F0 新增 15 个关系为 7 TP/8 FP，仍比 rules 低 0.001796779 Score。这不是全面优于 rules 的结论，也不授权 dev/GPU/第四组合。

F1 为严格 train32→原80输入/16文档/39患者/1189实体/587候选/1557对；F0 为严格 train48→原58/16文档/47患者/947实体/415候选/1166对。各模型/候选来源沿已有隔离证据，实际核患者优先训练/留出文献 ID 无交叠，输入固定；无新 fit/模型运行。原实体、非关联字段及患者元数据全部精确保持，两折旧关联均未删除。

F0 患者优先来自独立 limited-codec recovery：57 complete/1 partial，缺 1 个患者键涉及 8 个候选对，只保留显式正关联。原 strict 失败且未评分的记录继续独立保留；本控制未补缺失答案。

原 source74cc/plan310b/source-plan-freeze8e5a 已在运行前冻结。首轮 0.114597 秒在新预测/gold/评测前失败：helper 对 phenotype 排序，而旧并集文件保留历史列表顺序。只读诊断两折 20/11 处顺序不同、关系集合差 0；空 assembled 和 stderr 原留。独立 order-guard-recovery 仅将该等价 guard 改成逐患者集合比较，元数据 guard、输入、字面并集、阈值和 evaluator 均未变；恢复上限59秒，使两次执行总cap<60秒。source.diff 保留精确单行修改。

恢复 source734a/planaa5c/entry198c 在运行前冻结；两折 fullpred 共同冻结 2c15823d 后写 before-gold9de5ece5，再各调用一次公共 v2 all/text。旧 old∪reverse 和 rules 指标全部复用，0 次重评；共 2 次新评测。恢复实际 wrapper0.364852秒，主程序0.283060秒；两次 wrapper合计0.479448秒，均 CPU，无MCP/GPU/Git/官方上传。

完整产物在 order-guard-recovery/；全预测/日志留仓库外。source/plan、错误与恢复、共同freeze、格式/metrics/results 和最终 manifest 可作为一组交 validator 独审。结果当前为 pass_fixed_twofold_gate_pending_review_no_dev。
