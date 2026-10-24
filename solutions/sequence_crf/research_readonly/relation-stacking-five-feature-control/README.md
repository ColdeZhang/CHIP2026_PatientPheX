# 固定五源低容量关系融合：关闭此配置

唯一固定控制已完成。F0 Score **0.697656524607**，micro **0.695252679939**，macro **0.642696528524**。相对旧 SFT48 小幅提高，但低于预定的更强对照，因此不保留、不扫描、不迁移 dev。只是否定本次冻结配置，不能概括为所有 stacking 无效。

| 复用对照 | 原 Score | 新 Score 差值 | 新 macro 差值 |
| --- | ---: | ---: | ---: |
| oldSFT48 | 0.695443947581 | +0.002212577026 | +0.003673964041 |
| oldSFT48 ∪ reverse | 0.704687440059 | −0.007030915452 | −0.024404241894 |
| 同池规则 | 0.711353521870 | −0.013696997263 | −0.045620779488 |

保留门槛事先冻结为 Score 严格大于三者最大值0.711353521870099，且 macro 不低于三者最大值0.6883173080126084；两项均未满足。旧指标复用，只有一次新公共v2 all/text评分。

## 固定方法与实际范围

X严格为五个原[0,1]值：旧Qwen9B集合选择、同adapter逆序集合选择、binary32原概率、旧LR实体出现概率的max、rule_selected_mean候选出现投票均值。不加logit、标准化、交互或上下文，不把规则均值冒称校准概率。单一 sklearn LogisticRegression L2、C=1、lbfgs、max_iter=1000、tol=1e-8、class_weight=None、截距开启；固定p>=.5，未扫描或用expectedF1。

F1 1557行用于元训练，**明确打开并读取了含target的旧表**，仅投影指定特征与原target；365正例，保留原完整HPO复合串拆分后的字面标签，不改成另一个canonical366口径。F0 1166纯输入全key覆盖。F1不是元验证；F0已有反复开发历史，不是盲测。其他基模型/候选32→F1、48→F0来源复用已核证据，binary在两折始终是同一个只训练F2F3的train32模型。逆序来自同adapter，不宣称独立基座。

旧303列HGB和两列LR已做且低于强基线；本次是特定五源低容量组合，不是首次stacking，也不是纯单变量比较。此前只读准备的“不允许打开混合target表”由root这次明确授权取代，历史记录未改写。

## 执行与验证

source/plan在打开F1混合表和fit前冻结。实际一次fit/predict/模型保存回载及F0冻结，共约1.098秒，低于60秒；实际fit0.003885秒、19次迭代，无收敛警告。sklearn1.8.0有penalty参数弃用提示，配置未改变。模型仅5系数+1截距；保存回载的全部1166概率np.array_equal。

F0固定选274对。16文献、47患者、947实体完整；仅替换association值，实体/患者/其余字段在每行的原始UTF-8字节严格不变。完整预测和模型SHA先freeze，之后才首次读取F0 gold。公共格式校验0错误、0警告。

第一次评测在加载evaluate.py时，因同目录validate_predictions尚未注册而退出；发生于F0 gold及任何评分之前。原source/plan/reservation、真实trace转录与模型/预测保留在evaluation-attempt1-import-error及原目录。独立evaluation-recovery只先注册固定validator，再加载同一个固定evaluator；没有重新fit、预测、改变门槛或产生第二个新评分。终点在evaluation-recovery/terminal/results.json。

读取main AGENTS/HANDOFF/WORKLOG后仅在本外部目录写入，无Git、MCP、GPU或R1文件修改。真实入口/耗时/错误见execution-receipts.json；全量矩阵、概率、模型和预测留本目录，根级汇总只需小manifest/source/results。没有后续配置或dev迁移。
