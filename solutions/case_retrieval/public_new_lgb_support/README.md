# public-new 缓存关系：双折 LGB 支持控制

目标是验证现成 Task1 支持流程升级能否改善同一批 public-new 关系，不重新训练或推断神经模型。源、[计划](plan.json)和[来源合同](source-lineage.json)先以79e2ef5发布，唯一0.316秒CPU控制已完成，**两折keep=true，独立终审pending**；见[结果](RESULT.md)。小型检查与精确产物入口见 [指针](entry-pointer.json)。此前仅 F0 的库存建议由本次明确的双折开发控制取代。

## 固定变量与主终点

分别从旧 SFT 模板出发，把已冻结的 public-new raw 正关系与对应 fold 的现成 LGB nested 文档-HPO 支持集合相交。保留全部旧实体、旧关系的原顺序与值、患者信息；只替换顶层 `association` 值，其外 UTF-8 字节精确保留。不加入 raw 中没有的关系，不用新 Task1 实体覆盖输出实体。

| 内部折 | 旧模板/患者 | 固定 raw 正关系 | 旧支持新增数 | 旧支持 Score / macro |
|---|---:|---:|---:|---:|
| F0 | SFT48，947实体/276旧关系/47患者 | 112 | 30 | .7057171476841083 / .6548632336181232 |
| F1 | SFT32，1189实体/383旧关系/39患者 | 193 | 65 | .6821288043976663 / .5883696702317567 |

双折完整预测共同冻结后，每折只调用一次公共 v2 `evaluate_report(all,text)`，旧指标直接复用。其内置固定敏感性诊断照常报告，只有默认口径判定。**两折各自 Score 严格高于旧支持，且各自 macro 均不下降，才 keep**；否则关闭，不补阈值、并集、fallback 或单折择优，不自动迁移原dev。

## 来源与解释边界

- F0 支持模型 train48=F1+F2+F3；F1 支持模型 train48=F0+F2+F3。各自的训练标签、嵌套候选生成器、held 输入 baseline/Sap detector/CRF 都排除完整目标折。固定公开 PubMedBERT/Bioformer/BioBERT 只对纯输入推断，没有项目任务拟合；全折 span whitelist 是旧池重建旁路产物，不进入当前 BioBERT 合并。
- **F1 旧、新支持都是 train48**；保留 SFT32 raw 不代表整链 strict32。F1支持训练见过F0，不能把后续F0称作整个流程从未使用的独立 outer test。
- cut .3 曾在 pooled64 的九个阈值中选择，LGB方法选型也已看内部折。这是训练祖先排目标折的开发控制，不是全流程未用目标折标签。公开预训练也不声称排除了论文原文。
- F0旧HGB96/greedy→LGB102，F1旧HGB95/greedy→LGB101，并增加BioBERT特征/候选和nested解码；所以结论归于支持流程升级，不能归因为仅更换树算法。raw、SFT适配器、模板固定，不使用 public212 新模型。

## 执行与产物

入口：`python solutions/case_retrieval/public_new_lgb_support/run_once.py --plan solutions/case_retrieval/public_new_lgb_support/plan.json`。已唯一执行完成，不重启。子进程总上限30秒，TERM后1秒强制结束自有进程组；所有实际输出写计划中的仓库外 `actual-v1/`。目录已存在即停止，不重跑半成品评分。

预期只有两份完整新预测、共同冻结、两份完整指标与格式报告和一个 keep 结论；fit/model-predict/NN/GPU/原dev/官方调用均为0。首次失败保留日志和已有证据，无预算或方法重试。先运行的合成自检只验证字节保留、旧值保留、复合HPO支持、未知患者与重复字段拒绝，没有读真实预测或gold。
