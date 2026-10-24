# HGB → LightGBM：静态映射与有限历史核查

2026-10-24。仅读原训练源码、小型已完成manifest、现成sklearn包源码和文本历史；无train/gold/pool数据读取、fit、predict、GPU、安装或联网下载。本文件供路线负责人合入固定计划，不是运行结果。

## 既有实验与本次可检验变量

有限检索本体/序列/检索当前路线的 `.py/.md/*plan*.json/*manifest*.json` 未发现“同强BioBERT八池＋原nested政策”的LightGBM实跑。找到的LightGBM提及是官方Agriv方法说明与R1可行性动机，不是本任务模型终点；不据此声称已穷尽每个历史文件。

实际旧源 `public_phenotagger_biobert_selector/fit_control.py` SHA `ae870b823abaae2cc50809e633f9b4f5b0ecf13a5515a846e022b284d4309253`，通过相同 `feature()` 生成dense DictVectorizer矩阵；每折48篇/16篇，原生HGB160。小manifest确认四折维数102/101/103/103、训练候选46733/45403/45907/46480、held候选15028/16595/15309/15349。四fit实际15.021/13.317/13.405/12.807秒，共54.550秒；这仅是旧HGB来源实测，不是LightGBM耗时承诺。

## 最贴近当前源的固定参数映射

现成sklearn1.8.0 `_hist_gradient_boosting/utils.py:get_equivalent_estimator` 已给出正式测试用映射，比依赖LightGBM缺省值更可靠。以下名称是拟议LGBM wrapper写法，安装前仍由父线程按固定4.6.0源码核接口：

| 原HGB实际值 | 拟议LGBM参数 | 限定 |
|---|---|---|
| loss=`log_loss`，二分类 | objective=`binary` | 1棵二叉分类树/轮；不改正负权重 |
| max_iter=160 | n_estimators=160 | no early-stopping callbacks，不传验证集用于停止 |
| max_leaf_nodes=15 | num_leaves=15 | 都为best-first；不能保证每轮恰有15叶 |
| max_depth=None | max_depth=-1 | 明确无深度上限，避免None接口差异 |
| learning_rate=.06 | learning_rate=.06 | 同缩减值 |
| min_samples_leaf=35 | min_child_samples=35 | 对应native min_data_in_leaf；不要同时传两别名 |
| l2_regularization=5 | reg_lambda=5 | 同Newton叶分母的L2意图，非数值恒等保证 |
| 无L1 | reg_alpha=0 | 显式冻结 |
| max_bins=255 | max_bin=255、min_data_in_bin=1 | 后者来自sklearn对照映射，不能默认LGB的3 |
| grower最小Hessian=1e−3 | min_child_weight=1e−3 | 对应min_sum_hessian_in_leaf；同样不用双别名 |
| min_gain_to_split=0 | min_split_gain=0 | 固定增益下限 |
| max_features=1 | colsample_bytree=1、feature_fraction_bynode=1 | 不抽样特征 |
| 全样本，无bagging | subsample=1、subsample_freq=0 | 不加GOSS/DART，固定boosting_type=`gbdt` |
| 初始loss最优常数 | boost_from_average=True | 二分类对应logit先验意图 |
| 无特征bundle | enable_bundle=False | sklearn显式映射，兼顾feature顺序 |
| BinMapper subsample=200000 | subsample_for_bin=200000 | 当前每折不到47000行，均可用全部训练行做分箱 |
| random_state=20261023，OMP1 | random_state=20261023、n_jobs=1 | 固定data_random_seed同seed；确定性可设deterministic=True/force_col_wise=True |
| class_weight=None | class_weight=None、is_unbalance=False、scale_pos_weight=1 | 同原未加权目标 |
| dense numeric matrix | categorical_feature=[]；zero_as_missing=False | 不把数值零/one-hot误当missing，不引入类别原生切分 |

这些参数只定义一个近似匹配的算法替换，不要求运行一组超参候选。沿原八池、原feature/label函数、每折实际feature顺序、cut=.3和nested解码；不得附带新pool、重校准阈值或新后处理。

## 无法一一对齐的实现差异

1. **分箱**：HGB 255指非缺失箱，另保留missing箱；连续值超过255个时按`np.percentile(..., method='midpoint')`，否则取相邻distinct值中点。两库同max_bin仍不保证边界/空箱/零值处理完全一致。sklearn自己的`test_compare_lightgbm.py`为消除此差异会预先分箱；本控制应保留原输入矩阵，不额外预bin，否则又换变量。
2. **叶样本数与Hessian门**：HGB直接在分裂候选检查样本数和sum Hessian；LGB的min_data_in_leaf语义须按4.6.0实际源码/文档确认，不能仅凭同名值保证每叶恰≥35。原HGB叶值源码是 `-sum_gradient/(sum_hessian+l2+1e-15)`；同L2值不消除分裂统计和精度差异。
3. **生长/并列/精度**：HGB已是best-first，不能把本替换包装成“深度优先改叶子优先”。实际split gain排序、相同增益tie、直方图累积和阈值比较可不同；sklearn对照测试也明确保留此边界。相同seed不代表同随机流或树结构。
4. **初始bias与训练终止**：binary loss常数初始化的数学意图可匹配，但clipping和实现细节可能不同。n_estimators=160是上限；LGB若无合法分裂提前结束，应记录actual_trees/current_iteration，不能为了声称160而静默改门槛。
5. **确定性和内部预过滤**：单线程/固定seed可降低运行差异，不能跨库保证相同预测。LGB可能做feature_pre_filter、稀疏优化和特征顺序调整；固定计划应明确此类参数，尤其禁bundle，不按结果再调。

## 已核现成本地证据

- default和sequence-crf `.venv` 均Python3.12.14/实际sklearn1.8.0；`find_spec('lightgbm') is None`，distribution未安装。未找到现成本地LGBM源码；父线程另核固定上游资料，当前不宣称4.6.0实际已可import。
- sklearn `utils.py`：`bd22fb30a6068e15b77741d16dfc48eb108be7e2f455c0115d3e54695d0f5d0c`；`test_compare_lightgbm.py`：`d2e3e0afe97b5ea9a3683ff2c3f15438ba2e41b5850ddeb6c2e00f46d6318cb5`。`binning.py`（`7fb35b1e`）、`grower.py`（`008241aa`）、`gradient_boosting.py`（`5ccdc2f1`）与` splitting.pyx`（`1ea07a80`）均来自当前实际site-packages，完整SHA可由父线程最终manifest绑定。
- 依赖选择与正式运行成本由父线程汇总，本文件不授予fit/安装授权，也不作模型收益预测。
