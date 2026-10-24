# LightGBM full64/dev：只读来源核查

2026-10-24。仅核源码、已完成manifest/任务状态和既有审查；未打开train/dev gold、训练/输入pool正文、完整预测或pickle，未fit/predict/score。`T=/workspace/patientphex-runs/round2-20261023/ontology_rules/public-biobert-dev-transfer`。

**最小替换可复用已冻full64与dev两个pool、原feature函数及旧103列vectorizer；只fresh训练LGB模型。无需重新生成BioBERT候选、重新运行旧HGB或读取新的神经缓存。** 旧矩阵未落盘，因此只能称恢复相同特征配方和列顺序，不能声称比过旧矩阵的历史byte SHA。

## 准确输入与列来源

| 用途 | 既有位置（相对T） | SHA前缀 / 实际范围 |
|---|---|---|
| full64准备 | `prepared-train/manifest.json` | `4074c909`，64篇、62281候选 |
| full64训练pool | `prepared-train/train.pool.jsonl.gz` | `0b3d835a1946fcd40388f7a919d0f44731dbc773a3c16a3415515f261fcb8129`，3501710B |
| 旧full64模型manifest | `fitted/manifest.json` | `5dcd9748`，103列、HGB160 |
| 旧model/vectorizer bundle | `fitted/model-vectorizer.pkl` | `42e941150d1974993ea403ad41af6bfbabe410fe334f1c90c50a895d0a44c9a7`，298059B |
| dev纯输入pool manifest | `predicted/manifest.json` | `1c816bb6`，16篇、13246候选 |
| dev已合并纯pool | `predicted/input.pool.jsonl.gz` | `18ac347e1760d960d760de873fe16882525bcee937e2ceceb4a03b163c3c4ee9`，754726B |
| 旧dev完整概率 | `predicted/selector-scores.jsonl` | `3a38d1db`，仅作旧来源依据，不拿其概率当新LGB概率 |

full64不是拼接四个train48池。`prepare_train.py`（`fb1ccf3d`）按旧full64训练manifest的顺序拼接强B四fold各自的held16/input池；每个上游生成器训练48且排除对应16，四fold合64。候选字段保持，仅`origin_index`从held的None改为其OOF fold编号。新pool来源即强B八池manifest `0ba8dffd` 的四个input侧；旧full64是61781行，BioBERT增量后62281行。沿已有审查，不重新构造其他候选。

`fit_full64.py`（`73b8114d`）读取上述pool，以原 `fit_control.feature(multi,row,c)`（整个源`ae870b82`）构造矩阵；标签沿 `prior.labels`（fit_helpers `e8bac3d4`）。该feature仅取 `multi.features(row['input'],c,row['base'])` 加12个public/Bioformer字段及6个BioBERT字段。`multi`及其依赖由旧full64/reference manifest绑定，不能用现在另一个模块版本替代。

原 `DictVectorizer(sparse=False)` 已与model一起保存，bundle含`model/vectorizer/features`三项。fitted与predicted两个manifest的103个feature名称及顺序实际完全相等，紧凑JSON（ensure_ascii=False,separators=(',',':')）SHA为 `0371207897da6750bc0c7e30e7f195e0a1466ee7263572fed5f944bae386df5b`；旧97列加6个BioBERT字段。最小新入口可校验bundle SHA后只使用原vec.transform，断言feature_names_/vocabulary与manifest103列一致，不重fit向量器、不让dev扩词表；fresh LGB不继承旧HGB学习状态。此处未加载pickle或实际transform，运行入口仍需其真实完整性检查。

## dev缓存已经构成完整输入

`predict_dev.py`（`94df40e0`）原来从旧Bioformer dev概率池`c032ca68`删掉selector_probability，结合原pure dev `c0fcad36`和reference base，再按原`merge()`机械加入native BioBERT `.95`、每句ngram2–8的canonical/provenance。已完成manifest记录13144→13246候选、旧字段/类型/顺序保持；最终 `predicted/input.pool.jsonl.gz`已存下所有这些输入，无需重做merge或native推断。

缓存依据在 `R/research/public-phenotagger-biobert-dev16-cache/terminal/native-dev16-final-cache/`：manifest`0b28b1f8`、canonical`bd084a93`、provenance`3f3e961f`；全部实际引用见predict_dev-plan `8128b788`。新模型只读最终pool，旧概率不要作为额外特征带入。原`feature()`接收的row已有input/base/candidates，适合直接复用相同列构造。

## a179 模板与原实体链

`T/terminal-attempt2/complete-freeze.json`（`a84d2ea8`）把scores/model绑定到 `biobert.nested.jsonl` **baac8186**；该输出有1072实体。当前关系增强的 `R/research/reverse-json-candidate-order-dev/formal/assembled/freeze.json`（`7d80529e`）明确baseline就是baac，union输出为 **a179a5de5b7271979a0fba851afc269d63125248a56681674e7388d0c6728c39**。

既有 `validation/reverse-json-candidate-order-dev-review/review.json`（`5a106306`）确认1072实体及所有非association对象精确保持，旧368关系全留并新增23，a179为391关系。这里沿用已有审查，没有再读fullpred或重评分。新Task1输出应以该a179原始行做entities值替换，保持391关系及其余非entity UTF8片段原字节；不要退回旧d531/baac关系模板，也不要因新实体重新计算关系支持。decoder继续原greedy .3加 `decode_nested_selector.py`（`7f4484fc`），不采用已经关闭的全局兼容解码。

## 旧耗时与实施边界

旧full64 HGB实际训练入口14.706秒、wrapper15.084秒；原dev构池＋load＋predict3.305秒、wrapper3.524秒，都是rc0。训练矩阵若float64为62281×103（约51.32MB），dev为13246×103（约10.91MB）；这是矩阵字节推算，非进程峰值。新LGB≤120秒仍须以本次真实耗时为准，不能用旧HGB秒数保证。

不要直接重新执行历史plan：输出目录不可变，且旧predict_dev-plan仍绑定finalizer `b80b4044`，后续attempt2实际完成使用 `d5b3cdda`；无关finalizer旧SHA可令整套`sources()`失败。新wrapper应冻结实际复用的fit/feature/decode依赖和新入口，保留历史计划原字节，不为复用而改旧manifest。旧fit-plan`e66e75a5`/predict-plan`8128b788`是来源记录，唯一新full64 fit＋落盘回载＋新dev概率须落本路线独立目录。
