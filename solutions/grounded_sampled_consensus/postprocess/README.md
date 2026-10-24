# 双采样固定后处理

本目录只处理两个完整种子（20261024/20261025，各 F1 39 患者）。唯一质量端点为 **B4 ∪ (S1 ∩ S2)**，B4 固定为 524 条关系、Score `.701325377212172`。每条流先独立解释原文 quote、按同一 train32 别名及 b59 公共规范化政策映射，再按 `(文献, 患者, 类型, literal值)` 取交集。不能在规范化前交集词面，不能按基底差集挑样，也没有单种子质量端点。

复用原 `interpret`、`grounded`、`StrictNormalizer`、`semantic.decide` 和 `new_relations` 源码；`common.py` 逐 SHA 固定。公共 41099 bank 与 encoder 由父线程的原查询入口运行。唯一变动是容纳两条独立流，以及对完全相同词面的必要查询去重；两条流的患者、重复 occurrence、定位、原始拒绝原因和最终映射各自保留。presence 已在 GPU 前明确固定为 0，原 1.5 提议未执行。

所有阶段计划先保存/推送，再执行。以下命令工作目录均可为本目录；全量产物默认放 `R/research/grounded-sampled-consensus-v1`。

1. `python bind.py template` 保存未绑定模板。`python contracts.py --output <外部路径>` 只做合成边界检查，不读任何 held 内容。
2. 完整原 78 条 collection 到齐后：`python bind.py extract --collection <collection.json> --runner <实际run.py> --inference-plan <实际infer-plan.json>`；随后 `python process.py extract --plan extract-plan.json`。
3. `python bind.py prepare`；`python process.py prepare --plan prepare-plan.json`。输出 `normalization-preparation-v1/manifest.json`、`queries.json` 和全部两流 `strict-decisions.json`。父线程只对这个完整 queries.json 运行已有 Sap 查询；不重编码 bank。
4. 全查询 collection 到齐后：`python bind.py finalize --query-collection <完整query collection.json>`；`python process.py finalize --plan finalize-plan.json`。若必要查询为 0，省略参数。
5. `python bind.py quality`；`python quality.py assemble --plan quality-plan.json`，完整预测/来源/两流所有关系键/交集共同 freeze。
6. `python quality.py evaluate --plan quality-plan.json` 为唯一 30+TERM1 worker。若新增为 0，断言预测与 B4 逐字节一致，复用旧指标且 0 v2；否则只新增端点 1 v2，旧 0。任何已存在 launch/intent 不自动重试。严格 Score 增加才保留，macro 完整披露不设额外门。

原 dev gold 始终禁读。F1 gold 仅评分 worker 在完整固定 freeze 后开启；本目录编写/合成验证阶段没有执行真实解析、规范化、推断或质量。
