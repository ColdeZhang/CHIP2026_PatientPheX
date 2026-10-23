# 公开 Bioformer 单词候选固定追加：负结案

全输入机械生成的单词候选，在当前强基底上直接追加使四折均下降。pooled Score **.727975656 → .716642231（−.011333424）**；391个新增实体只有17TP、374FP。按预先标准关闭本配置，不换阈值或基底、不另拟合HGB、不迁移原dev。缓存完整、来源可核，不代表其实体可直接采用。

## 固定假设和协议

假设为原神经生成器排除1gram使部分跨度无供给，增加统一单词输入可能补召回。只对全部train64四字段原文启用原不可达1gram分支，沿原tokenizer/pun/POS过滤；不使用89个gold错误选词。原排除单词的作者意图未核实，本项是政策扩展。准备细节见[普查记录](P-20261023-PUBLIC-PHENO-UNIGRAM.md)。

公开Bioformer单模型原H5/head/tokenizer不变，CPU前向13145不同lower词串（复用探针256），原top1非None、native五位舍入后严格>.95、固定current-HPO映射产生529词/2516坐标。不新增hybrid、缩写或祖先扩展，不改变旧2–8/dict缓存。

唯一基底为已审 `selector-nested-control` 四折，保留4510旧entity对象，含此前新增的195个nested实体；所有非entities对象/关系完全相同。使用旧否定和非交叠追加，任何与旧实体重叠或包含均拒绝，候选排序仍为score降序、length降序、offset升序、HPO字典序。实际入口/计划在提交 `ee008b3` 先推送，再运行；四完整预测共同freeze后才读gold并运行v2。四折属于已开发训练侧比较，不是独立泛化或原dev达标。

预定保留条件为pooled两实体F1均值严格提高，且至少3/4折该目标不降。实际两条均不满足，四折全负。

## 结果

| pooled指标 | 原nested基底 | 固定追加 | 差值 |
|---|---:|---:|---:|
| mention F1 | .766299111 | .739005343 | −.027293768 |
| document F1 | .803367149 | .785327220 | −.018039929 |
| relation micro F1 | .687846109 | .687846109 | 0 |
| relation macro F1 | .654390253 | .654390253 | 0 |
| Score | .727975656 | .716642231 | −.011333424 |

2516候选中2007因旧实体重叠、118因否定拒绝；追加391个，mention增加17TP/374FP，document增加23TP/142FP。旧entity对象逐项保留，全部非entities exact，格式0错0警。

| 折 | 新增实体 | 新增mention TP/FP | 实体均值差值 | Score差值 |
|---|---:|---:|---:|---:|
| F0 | 130 | 0 / 130 | −.036388324 | −.018194162 |
| F1 | 91 | 6 / 85 | −.017663480 | −.008831740 |
| F2 | 96 | 3 / 93 | −.024021221 | −.012010611 |
| F3 | 74 | 8 / 66 | −.013379103 | −.006689551 |

因此“完整词串缓存＋高原生置信度”不足以保证临床任务中的实体精度。本结果只否定当前无上下文单词直接追加配置，不据此宣称所有单词边界建模无效；不从gold错例生成补丁。

## 运行、核验和恢复

全量CPU缓存外层239.276秒、程序238.288秒，其中101批新前向221.986秒，均低于480秒硬限。CUDA隐藏、实际GPU设备为空，TF两线程。唯一质量端点程序.805秒/外层.868秒，低于60秒；所有任务正常结束。

准备独审 `d9121489`、CPU256独审 `88ba241b`、质量独审 `379f2c4b` 均通过。质量审独立重装四预测及pooled逐字节相同，五组v2完整字段一致，确认先freeze后gold。全量cache独审 `751a5262` 逐项重建13145键/256复用字节/529词门控与2516完整sidecar，全部通过；不重复NN/H5/tokenizer。

源码在 [public_phenotagger_unigram_preparation](../public_phenotagger_unigram_preparation/README.md)，实际路径/SHA/大小、逐折分项和审查见[结果索引](../evidence/round2/public-phenotagger-unigram-results.json)。外存 `ontology_rules/public-phenotagger-unigram-preparation/` 的 `full-cache/terminal/output` 保留6.45MB原始top3和1.58MB sidecar，`control/terminal` 保留四预测/冻结/分项；Git仅源码、计划与小索引。已有缓存不得重复下载或NN。本实验关闭后不修改这些终点。
