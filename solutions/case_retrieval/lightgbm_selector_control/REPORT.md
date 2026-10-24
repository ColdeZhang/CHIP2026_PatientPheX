# 强 BioBERT 选择器的固定 LightGBM 控制

2026-10-24；当前状态：来源、私有依赖、原生接口与唯一计划已冻结，正式四fold尚未开始。目标是检验成熟树实现的具体边际收益，不重造候选/调阈值，不宣称更换库保证提分。

## 依据与唯一变量

官方overview §4.2 Agriv确实使用两阶段LightGBM选表型—HPO，并另用LightGBM融合关系证据。当前强选择器实际是sklearn HGB。有限检查三路线源码/实验记录未发现“同强BioBERT八池、同nested政策”的LightGBM实跑；已出现的名字主要是官方方案引用。此控制仅替换Task1的一个二分类选择器，不是冠军两阶段排序系统的复现。

严格复用强B `prepared/manifest.json` 0ba8dffd：每折48训练/16输入，训练候选46,733/45,403/45,907/46,480，held候选15,028/16,595/15,309/15,349；原feature/label函数、候选对象与顺序、训练侧资源隔离、dense DictVectorizer保持，实际列数102/101/103/103。新fit在运行时对照旧强B的feature顺序、训练数/正例数；不重fit旧HGB。cut=.3与原greedy后proper-nested追加完全不动，刚关闭的全局兼容解码不进入本控制。

现成sklearn1.8 `get_equivalent_estimator` 提供映射：160轮、15叶、min35、lr.06、L2=5、max_bin255，另明确min_data_in_bin1、Hessian下限.001、boost_from_average=True、禁bundle、分箱取样200000。无bagging/GOSS/DART、原未加权binary目标，固定seed20261023/单CPU线程；不引入原生类别切分，dense ndarray沿LGB auto处理为数值列。禁feature_pre_filter、零不作missing，固定确定性col-wise。所有参数见[configuration.py](configuration.py)，映射与不可等价边界见[静态核查](STATIC-MAPPING.md)。

两库本来都能best-first生长，不能将本次描述成首度叶优先。max_bin数值相同不保证边界相同；样本/Hessian门、初始bias、统计精度、分裂并列和树停止规则仍可能不同。不预先离散原矩阵以追求两库一致；不按结果调参数。实际不足160轮或接口/来源不兼容即停，不能改门槛冒称匹配。

## 依赖与实际接口

开始时default、sequence、semantic、fuzzy四本地环境均无LightGBM，pip缓存无wheel。root另行授权后下载唯一CPU wheel `lightgbm-4.6.0-py3-none-manylinux_2_28_x86_64.whl`：3,569,831B，SHA `cb19b5af`；ZIP有效载荷10,592,451B。`--no-deps --target`安装至外部 `research/lightgbm-selector-readiness/deps/runtime`，耗时1.619秒，无global/shared包改动、无模型下载。沿现有Python3.12.14/sklearn1.8/numpy2.3.5/scipy1.17，glibc2.41/libgomp现成。

实际1.163秒synthetic-only接口：128×4人造矩阵、原160轮配置、classes=[0,1]；正概率列与native booster完全相同，概率有限/逐行和1，pickle保存回载概率exact。已逐文件核安装包与固定wheel字节一致，native lib SHA `6a12e1a7`，ldd无缺库。另直接调用native参数别名表，所有实际传入键均被识别。两条sklearn“ndarray无feature names”警告如实保留在外部正常输出；实际列序由同vectorizer及回载对照验证，不按警告改变输入。未在接口阶段读任务数据或标签。

## 事前冻结的执行和判定

唯一4fold新fit/predict共享180秒，含原数据检查、特征构造、模型保存及全部held概率回载对比；首失败停，不换版本、参数或预算。原源码只改estimator及LightGBM实际状态校验，差异见[fit-fold-source.diff](fit-fold-source.diff)。完整四fold新概率/预测先共冻，全部非entity原始UTF8片段及关联保持，之后各fold一次和pooled一次原v2 all/text；旧强B指标复用。decode/eval另固定30秒CPU。

保留条件：pooled Score严格提高且至少3/4fold Score不降。候选与关系固定，所以变化来自实体选择；仍是曾用于阈值与方法研发的train64分组开发控制，不称盲测或原dev达标。强对照pooled **.730993616492**。本轮不执行full64/原dev迁移或官方提交，后续由root据结果决定。

入口[formal-plan.json](formal-plan.json) `d8935e74`，[entry-manifest.json](entry-manifest.json) `59aaa484`；外部根 `R/research/lightgbm-selector-readiness`。正式恢复查看外部 `formal-control/execution-state.json`，不要重复启动已存在输出目录。唯一执行命令为计划python运行本目录 `execute_control.py --plan <外部formal-plan.json>`。模型、全量概率/预测、正常日志保持仓库外；本目录仅源、计划、实际接口和关键终点。
