# NCR 替代后的 fresh HGB：静态实施计划

仅定位现有源码、manifest 与已审依赖；未读 gold/池内容，未合并候选、fit、预测、评分或访问 MCP。本文不阻塞正在进行的无标签缓存，也不表示后继训练已执行。

路径约定：`W=/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules`，`B=/workspace/patientphex-runs/round2-20261023`，`N=B/research/public-neuralcr-readiness`。详细旧依赖见 `N/cache-replacement-static/feature-dependencies.{md,json}`；八池总 manifest 是 `B/ontology_rules/public-phenotagger-top3-control/old-pools/manifest.json`，SHA `87264eb85a1801e249ac6f052faeec5ddeb03fa3044ecb014ca3b9c8d6d220bf`。

## 1. 新池与两个真实来源特征

从上述 `fold0..3/{train,input}.pool.jsonl.gz` 直接重建，保留 semantic/base/CRF、PubMed、Bioformer，完全避开 BioBERT 增强池、六列、独有候选与旧 selector posterior。原池重建入口为 `W/public_phenotagger_top3_control/prepare_old_pools.py::main/build`（27、52 行）：`calibrate_multi.pool` → `select_public_candidates.merge_pool` → `select_public_bioformer.merge_bioformer`；现成八池可直接复用，无须重跑旧模型。

最小新字段固定为 `ncr_present: bool` 与 `ncr_score: float`。仅对完整 NCR native raw cache 经已冻结当前 HPO 映射后真实存在的精确 `(pmc_id,offset,length,identifier)` 键置 true；score 为该规范键实际神经 float32 分数的最大值，保留 raw HPO、cache row 与 score bits 作外部来源证据。缺源为 false/0，不能解释为模型对所有旧候选实际输出了零分。NCR 没有 dictionary/abbreviation 流程，不创建假来源字段。

NCR 先完成原生 1–7 词、严格 `>.8`、原重叠处理，再执行当前 HPO ID 映射；不能把 683438 个枚举跨度或 408272 个 token key 当预测实体。映射资源复用 `B/research/public-phenotagger-reuse/build_label_map.py`（SHA `1b4a60a8a965a4704a3a78e52fa56e628fc855cf0781ec7b9c32b5cb3cd57e03`）的当前 HPO 语义，绑定 `/workspace/patientphex-data/hp.obo` SHA `a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b`；新 NCR head 必须由自己的 14260+None 列生成映射，不能复用 PhenoTagger 17266 列映射。规范键合并不再次偷偷运行不同的跨度重叠算法。

可适配 `W/public_phenotagger_biobert_selector/prepare_pool.py::{defaults,same_old,merge}`（36、42、49 行），但需移除 BioBERT 权重/5903 键/2–8 词/.95/六字段/provenance 的硬编码。旧候选始终作为原顺序前缀，除两个新字段外逐值逐类型相同；共享键只附实际 NCR 字段；NCR 独有键以冻结 native 来源顺序追加，不能残留 BioBERT 独有键。

新键的缺失来源默认值沿原合并契约：`probability=.5, similarity=0., exact=False, is_base=False, crf=False, semantic_source=False`，前两模型 12 个字段缺源，`crf_span` 仅依据旧池同跨度真值；这些不是新 Sap/CRF 预测。保留原 `input/base/origin_index`，每个键核原文切片与唯一 passage。旧候选自带 semantic `probability` 仍保留；旧 `selector_probability` 不进入新特征。

## 2. 四折资源与训练边界

每折旧 fit manifest 都在 `B/sequence_crf/public-bioformer-selector-control/fold{k}/fitted/manifest.json`，且由八池 manifest 的 `old_fit_manifest` 完整 SHA 绑定。下表路径均相对 `B/ontology_rules/`；精确三内层输入/生成器/CRF 路径由各 calibration manifest 固定，不能自行拼一个看似对应的文件。

| 外折 k | 留出 | 训练内层 held16 | 旧 train/input 行数 | 旧特征列数 | 原 reference；training_manifest |
|---|---|---|---:|---:|---|
| 0 | F0 | F1、F2、F3 | 46349 / 14910 | 96 | `nested-crf/source-control-reproduction/pred.manifest.json`；`nested-crf/calibration-manifest.json` |
| 1 | F1 | F0、F2、F3 | 45039 / 16454 | 95 | `nested-outer1/source/pred.manifest.json`；`nested-outer1/calibration-manifest.json` |
| 2 | F2 | F0、F1、F3 | 45561 / 15154 | 97 | `nested-outer2/source/pred.manifest.json`；`nested-outer2/calibration-manifest.json` |
| 3 | F3 | F0、F1、F2 | 46064 / 15263 | 97 | `nested-outer3/source/pred.manifest.json`；`nested-outer3/calibration-manifest.json` |

Fj 的文档 ID 由 `/workspace/patientphex-data/round2-inner4/fold-j/validation16.input.jsonl` 与八池 manifest 的 `input_ids/nested_input_ids` 定义。对外折 k 的训练行 Fj（j≠k），base/semantic/CRF 生成器训练于 `train64 − Fk − Fj` 的 32 篇；对外折 held Fk，生成器只训练于互补 48 篇。fold0 内层生成器资源在 `nested-outer0-inner{j}/`，CRF 在 `nested-crf/inner{j}/`；其他折在 `nested-outer{k}/inner{j}/`。outer base 为 `fold{k}-baseline/pred.jsonl`，semantic 为 `frozen-fold{k}-sapbert/scored-candidates.jsonl.gz`；outer CRF：k=0 用 `nested-crf/outer0/pred.jsonl`，其余用 `nested-outer{k}/outer{k}/crf.jsonl`。

继续沿用 `W/public_phenotagger_biobert_selector/fit_control.py::main` 32–53 行的 provenance 检查：48/16 不交、三个 held16 不交且并为 train48、每个生成器32恰为相应补集、每项 input/base/candidate/CRF/生成器 SHA 与原 reference 一致。NCR/PubMed/Bioformer 仅公开固定权重的 input-only 前向可跨折共享，不能因此放宽任务训练生成器隔离。

**H40 边界**：上述池只对原四折目标成立。若改做新 train32→F1 等留出，逐文 OOF 并不保证资源未见 F1；必须重查所有候选、词典、门控与模型的训练 ID。不可简单裁切原八池/旧四折 posterior，也不可把 future full64 HGB 的训练内分数称留出结果。

## 3. fresh estimator 与实际产物

直接适配 `W/public_phenotagger_biobert_selector/fit_control.py::feature/main`（19、73–98 行）：特征是原 `calibrate_multi.features(input,candidate,base)` + 原 `public_*` 六列 + `bioformer_*` 六列 + **仅 `ncr_present/ncr_score`**。每个外折独立新建 `DictVectorizer(sparse=False)`，只对该折 train48 执行 `fit_transform`，held16 仅 transform；每折新建 HGB，不加载旧 pickle、树、向量器或 posterior，也不 warm-start。

原训练参数保持：`HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=35,l2_regularization=5.,learning_rate=.06,random_state=20261023,early_stopping=False)`。原 manifest 其余参数：`loss='log_loss', max_bins=255, max_depth=None, max_features=1., class_weight=None, categorical_features='from_dtype', warm_start=False`，以及默认 `interaction_cst/monotonic_cst=None, n_iter_no_change=10, scoring='loss', tol=1e-7, validation_fraction=.1, verbose=0`；`.1` 因 early_stopping=False 不触发随机验证。原运行 sklearn=1.8.0，运行前绑定实际版本与完整 `get_params()`；线程沿既有 CPU 单线程设置。

标签函数沿 `W/public_phenotagger_biobert_selector/dependencies/fit_helpers.py::labels`（70 行）；后续正式 fit 才读取相应 train48 gold，本静态阶段未调用。不要误调用该 helper 的旧 `check_pool/added_raw_features/main`：它们仍含已否决 prehybrid raw 特征。仅保留其被明确使用的加载、来源、标签辅助契约。

事前先冻结新八池的实际行数与 SHA，再定有据预算。旧 `maximum_training_row_ratio=1.5` 是上一控制的预算护栏，不是 NCR 语义；不能为过关按 gold/FN 修剪候选，不能照搬 BioBERT 246804/62281 等行数。继续执行原 selector `.3`（概率 `<.3` 才丢弃，即接受 `>=.3`）与原 nested 装配时应各自保存源/配置，不与 native NCR `>.8` 混淆，也不临时扫描阈值。

每折保存 fresh model/vectorizer 的 SHA、完整 feature_names、实际训练/held ID、来源 manifest、新旧行数、runtime、estimator、`n_iter_` 和树 state。**新 F（真实列数）、新增类别列、每树实际节点/叶子/阈值与数值状态数量均待真实 fit 后填写**；不能预写 F=旧列数+2，更不能沿用 103 列的旧树状态。新候选可能带来额外 section/ptype 类别；需断言最终无任何 `biobert_*`，实际新列差集与候选特征契约吻合。推断后才写新 posterior；质量结论另待授权完整终点。

以后如进行 full64→原 dev，入口参考 `W/public_phenotagger_biobert_dev_transfer/prepare_train.py::main` 与 `fit_full64.py::main`：拼四个新 held16 池，只把 `origin_index=None` 改为外折号；候选生成器各为互补48。旧基底总行数61781，新总行数须实际统计，移除旧62281硬编码；fresh full64 vectorizer/HGB，只用 train64。原 dev 的 NCR 缓存、候选、后验和实体装配仍是独立后继步骤，本计划未启动。

## 4. 保留 97 列关系支持 HGB；移除 103 列实体 HGB

已审关系闭环是 `B/validation/biobert-relation-dependency-review/review.json`，SHA `a09d7978567d305375ba241f8bac7b0bf495828cb23384f422bef54c2a61d0a2`。固定 public-new 支持来自 **BioBERT 加入前**的 PubMed/Bioformer 97 列 HGB：`B/sequence_crf/public-bioformer-dev-transfer/fitted/selector.pkl`，295637B，SHA `db2fc28850596bf633b1baf33a90d3747341d3d621f87dc29d87ad374a5bbd51`；manifest SHA `9c52eac7d20fb62f9fdf3c14f6b469a8df6830076d22d0abc5d0cc5333d3380d`。**它及前两公共模型仍是冻结关系支持的真实依赖，必须保留并计入最终模型/资源谱系**，不能因为新实体 HGB fresh-fit 就扣掉旧97列模型。

它的 `B/sequence_crf/public-bioformer-dev-transfer/calibrated-point3.jsonl`（SHA `6401a24e06ea48d8e8711313b7dfe726f6fa7a9dae4ec93e21e97e1707835fbd`）固定支持35条 public-new 关系；a179 的 association 投影及原患者/文档顺序可沿已审闭环复用，a179/Baac 的 BioBERT 实体不可复用。若把支持改为新 NCR 实体，须按新政策重算，不能继续声称原35条依赖不变。

应退出新实体链的是 `B/ontology_rules/public-biobert-dev-transfer/fitted/model-vectorizer.pkl`（103列，SHA `42e941150d1974993ea403ad41af6bfbabe410fe334f1c90c50a895d0a44c9a7`）及四折对应旧模型/后验、BioBERT 候选与六字段；文件原样保留作历史证据。只有实际替换闭环成立后才计算预算。`9,966,949,054` 仍只是已审神经组件替换的条件投影；新 F/树 state 未 fit 前不得宣称最终完整参数核算已完成，更不能将原97列关系支持模型漏算。
