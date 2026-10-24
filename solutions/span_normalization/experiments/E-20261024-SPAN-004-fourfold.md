# E-20261024-SPAN-004：固定 GP 特征四折补全

状态：源码/计划已冻结，两折CPU合同通过，GPU 未启动。原验证集未读，官方未提交。

已知 E002 F1 Score +.002115173201、E003 F0 −.001065881294 后，主线程授权补齐另两折，检验整体稳定性。这是研发补全，不是盲测；不改候选池/HPO、GP 八轮、LightGBM 160树、.3 cut、nested 解码或 missing 规则。成功门为四折 pooled Score 严格升且至少三折不下降；否则关闭此特征控制，不扫参数。

F2/F3 实际旧列均 103→105，训练行 45907/46480、正例3180/3207、held候选15309/15349。旧模型/全概率/完整预测/指标均复用，CPU仅核列投影和旧概率/解码一致，不重fit旧模型，不重评旧指标。

模型来源：F2 inner0/1分别复用GP13/03，inner3新训GP01；final新训GP01348。F3 inner0/1复用GP12/02，inner2复用本次GP01，final新训GP01248。四个旧GP32的原train文件SHA、32文献IDs、原八轮配置与本次来源全部相同；每组训练排各自目标16及outer16。原五个GP32中的GP23此次不需重跑。共三次fresh训练、八次全target score；包含低于−2的有效span，未对齐/无共享窗显式missing，全部原候选不筛。

预算依据：既有三模型/四pass实测277.219与272.023秒。本次估338秒，.3 GPU0唯一900秒+TERM30，排DocRE30真实release后。训练内实际梯度/坐标/完整窗口检查沿用已审固定入口；CPU部署重新核八组输入映射。禁止同卡并行或重复启动UNKNOWN。

入口：`build_oof_completion.py` 构建 `oof-completion-plan.json` 及F2/F3子计划；`prepare_oof_completion.py` 按每job outer隔离；`run_oof_completion_gpu.py` 顺序11step，复用新GP01而不重复训练；`fit_oof_completion.py` 每折仅fit+完整预测冻结；`evaluate_oof_completion.py` 同时冻结两新折及旧F0/F1完整预测，再各一次F2/F3 v2及唯一pooled64 v2。已有负F0仍进入pooled，不逐折挑模型。

大产物外置 `round2-20261023/stage8-span-normalization/oof-fourfold-completion/`。sources与全部来源SHA在计划；旧F1/F0源码保持原字节。既有F1独审bcb53dba、F0独审a3f855a0已通过，不将独审变成逐步骤审批。

CPU实际12.247/12.887秒，103旧列投影完全一致，15309/15349全部旧概率及16篇旧nested实体精确一致，零新fit/零v2。见`evidence/oof-fourfold-readiness.json`。

计划说明勘误（运行前）：F2/F3子计划的非执行字段`training_count_change`沿用了E003的“F1→F0”文字；真实`outer_fold`、全部32/48 IDs、train/input SHA、jobs及本记录均为本次F2/F3。保原冻结SHA与上传intent，解释字段不参与执行；不得按该旧文字误读划分。
