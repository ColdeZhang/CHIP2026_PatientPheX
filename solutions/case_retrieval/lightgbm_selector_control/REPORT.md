# 强 BioBERT 选择器的固定 LightGBM 控制

2026-10-24；当前状态：唯一四fold已完整结束，pooled小幅提高、3/4fold不降，满足事前保留条件；统一终审1d6ed9fd已通过。目标是检验成熟树实现的具体边际收益，不重造候选/调阈值，不宣称更换库保证提分。

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

## 唯一四fold实际结果

四fit/predict/完整保存回载共**59.389秒**（15.979/14.262/14.621/14.524秒），decode/eval **2.933秒**，5个子进程及controller均rc0；每个模型实际160轮、classes=[0,1]，全部held正列与native predict及pickle回载exact。没有重fit旧HGB。四份完整预测于04:48:50.913303共冻，eval marker为50.914316，之后新四fold/pooled各一次v2，旧指标复用。

| 指标 | 旧HGB | 固定LightGBM | 差值 |
|---|---:|---:|---:|
| pooled Score | .730993616492 | .731545429370 | +.000551812878 |
| F0 Score | .739700183730 | .740572140440 | +.000871956710 |
| F1 Score | .728056418176 | .728697536469 | +.000641118293 |
| F2 Score | .753373106038 | .754425066951 | +.001051960914 |
| F3 Score | .706780475185 | .706374759627 | −.000405715558 |

pooled menF1 **.771948494→.772867639**，docF1 **.809789609→.811077716**；关系micro/macro及计数完全不变。44篇实体选择变化，49加55删；全部非entity UTF8原片段保留，四fold格式0错0警。F3回撤保留，不包装成四折全正。**keep=true**仅表示满足该开发控制事前条件，效应很小，不是原dev提升或.8达标。

[结果](results/result.json) `6fc2c9e7`、[freeze](results/complete-freeze.json) `e31cfa67`、[完整终点清单](results/terminal-manifest.json) `087572cf`。入口独审[084923ce](results/entry-review.json)通过；[完整唯一终审1d6ed9fd](results/terminal-review.json)通过，未重fit/评分/读取新gold。root已据此另授权唯一full64/原dev迁移准备，本四fold源/质量保持冻结，不因后续修改。

计划继承旧模板的`status/runtime_checks/success`文字未及时清理，其中仍写HGB/120秒；这是未执行的旧描述字段。实际estimator、180秒数值门和Score+3/4条件已在新字段及controller中事前固定，实际运行也一致。原plan不改写；[明确说明](results/plan-interpretation-addendum.json) `c19fb7ea`逐项标出旧字段和真正执行入口，不将旧文字冒充实际配置。

## 独立的唯一full64/原dev迁移：源与计划就绪

四fold终审1d6ed9fd通过后，root授权一个同配置fresh full64模型。直接复用已冻结64篇62,281候选pool `0b3d835a`和16篇13,246候选纯dev pool `18ac347e`；从旧bundle `42e94115`只取固定103列vectorizer做transform，不重fit向量器或HGB，不重生候选/特征缓存。旧矩阵文件未保存，只能称同数据/源/列序，不冒称比较过历史矩阵字节。[完整来源](FULL64-LINEAGE.md)明确四held16合64的OOF链和a179继承baac的实体链。

新计划从最终作用域独立建立：[full64-plan.json](full64-plan.json) `745b1b85`、[entry](full64-entry.json) `c8c1d78a`、源[full64_dev.py](full64_dev.py) `e93a23c6`。单次fit＋完整保存回载＋dev predict，并连同解码/唯一v2全部放在120秒硬窗内，外层TERM5；没有沿用CV模板文字。新合并进程先用ontology feature，再显式给nested decoder绑定不同的sequence `run.py`；实际1.059秒纯导入合同证明原feature模块仍保留自己的baseline对象。未在准备阶段打开train/dev标签或pool正文。

唯一预测模板固定a179，**只替换实体**，原391关系（含既有35项public-new）、全部患者及非entity UTF8片段原样保持，不重算支持或加入正在训练的患者模型。完整新fullpred冻结后仅一次v2，旧 **.746081964986**指标复用，报告四项指标；仅Score严格升判为candidate best，≥.8另走复现验收。原dev长期用于开发，非盲测。sequence仍使用其独立a179关系端点，本控制不改其baseline，也无官方操作。实际当前尚未运行，恢复入口为外部 `lightgbm-selector-readiness/full64-dev-control/execution-state.json`，存在输出目录时禁止重复启动。
