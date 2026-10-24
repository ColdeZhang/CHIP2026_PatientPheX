# NCR fresh HGB 源码准备，尚未拟合

`fit_ncr_control.py --plan PLAN --fold 0..3` 仅对应旧控制的 fresh fit + held16概率阶段；输出 `fold{k}/fitted/{model-vectorizer.pkl,selector-scores.jsonl,tree-state.json,manifest.json}`。不生成训练posterior，不解码、不执行nested、不评分或扫cut。此阶段只是源码；`execution_ready` **只能在 root 明确授权实际拟合、冻结真实新池与预算计划后置true**，默认准备计划不启动它。

计划必需键：`execution_ready, sources, prepared_manifest, old_prepared_manifest, ontology_source, output_dir, cut=.3, folds`。`folds` 固定4项 `{fold,old_fit_manifest,reference,train_gold}`；文件ref用绝对path/SHA（有bytes则同核）。`sources` 为路径→SHA，包含新fit脚本、同目录 `prepare_ncr_pools.py`、冻结旧 `fit_helpers.py`（SHA `e8bac3d485074052175a5a365d750fd3d9fdb8d5b131260ba42bf10af6578aff`），以及原 `load_modules` 要求的 legacy public selector 和原 ontology feature modules。不要求旧1.5 row ratio；实际新池行数从已冻结prepared manifest读取。

继承原 `public_phenotagger_biobert_selector/fit_control.py` 的48/16与三条32→16验证：原old8池SHA、来源manifest、train/input完整SHA、内层gold/input/generator补集和原标签一致性、原candidate前缀/metadata/origin、原old positive count均保留。实际执行只读对应train48及其子集标签，held仅四字段输入，原dev不读。本次写源未打开任何gold/池内容。旧fit manifest只是来源/参数/feature metadata，不加载其模型pickle或posterior；源helper中的raw控制main/旧raw特征构造没有调用。

特征仍为原 `calibrate_multi.features` + `public_*`六项 + `bioformer_*`六项，新增仅 `ncr_present/ncr_score`。源和vectorizer层拒绝 `biobert_*`、`selector_probability`；原semantic candidate `probability`保留。每fold fresh `DictVectorizer(sparse=False).fit_transform(train48)`，held只transform；fresh HGB保持 `max_iter=160,max_leaf_nodes=15,min_samples_leaf=35,l2_regularization=5.,learning_rate=.06,random_state=20261023,early_stopping=False`，完整`get_params()`与原两模型old_fit/reference相等，warm_start默认false。实际库版本若旧manifest有记录则同核；完整runtime入新manifest。

F以真实vectorizer列数保存，不预写旧F+2；除两个NCR列外，新增列仅允许新候选实际带来的 `section=/ptype=` 类别。运行后核160实际迭代、无early stopping、二分类顺序、有限feature/probability；`.3`仅作为既定下游元数据记录，原nested由owner后继执行。

`tree-state.json` 在真实fit后读取全部实际 `_predictors`：逐树node/leaf/internal、全部value槽和真实leaf-value数量、存储threshold槽及实际有效numeric split数量、每个node字段（含结构整数）总槽数/实际数组字节/SHA、两类categorical bitset，以及baseline/classes/bin-mapper数组。核root可达、无循环/重复子节点、leaf≤15、实际F与bin mapper一致。manifest保留汇总+tree-state完整ref。**完整node结构的保守数值槽含非学习的结构整数，不冒充神经可训练参数；F、node/leaf/threshold/状态大小均不得提前虚填。** 模型pickle SHA另外覆盖实际序列化对象；数组计数明确不包含Python包装开销、字符串特征名与超参数标量。

文件独占创建；失败部分产物保留，不覆盖重试。此源码不改变原97列关系支持HGB的依赖；它继续单独保留，新NCR模型也不得加载旧103列实体HGB。这里只做内存编译/纯合成元数据检查，未执行实际fit、held预测或质量评估。
