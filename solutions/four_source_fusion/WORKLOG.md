# 四来源关系共识

2026-10-24 UTC。目标：检验新患者编码输出能否补强现有三源二票，不能只超过较弱单模型。

已固定[决定](decision.json)及[F1计划](plan-F1.json)，源码沿已核open-concept控制，仅将交集机制改为新C与三旧源至少一源一致。F1已完整装配及唯一v2，507→524关系（17新7TP10FP），Score .701087332488→**.701325377212**；micro+.001233、macro−.000281，实体两项不变。原dev最佳.755240不变。

source/plan62565701已先推送核验，13:04:07共同freeze后唯一1v2、旧0、无NN/fit。完整预测/指标在外部，见[入口](evidence/entry.json)。后续：等待原strict coded58，按原固定公式做F0；双折正才提出full64。不同去重解码政策不混入本实验。

自身12内容小包（含新完整预测）已在.100持久并实际2次read_file回读、逐SHA/fsync/reopen通过，zip31982B，回执[1f66ef85](evidence/F1-backup.json)。旧四源raw/模型沿原完整包继承，不重复备份。

F0原strict58已完整，按原决定绑定[同公式计划](plan-F0.json)。C单臂.700894已知、低当前强B3 .724457；此不是单臂过门替代融合验收。仍只在完整预测冻结后唯一v2，不改formula/source选择。

F0同公式完整正：**.728195628911**（强三源+.003738273579），13新增4TP9FP，micro−.002176/mac+.017129；947实体/47患者/16篇和330旧关系均保留。source-bound计划59622b87先推，13:17:19 freeze c1aeca22后唯一1v2、旧0，见[F0入口](evidence/F0-entry.json)。两开发折均正；允许准备fresh full64编码组件，不把coded单臂升级为更强单模型。

F0新12内容（完整pred+固定recipe/质量）小包27452B已.100实际2次read_file回读、逐SHA/fsync/reopen通过，[回执480f44c4](evidence/F0-backup.json)。F1旧包不重复，模型/原raw仍继承各源已核备份。

原dev源码准备：`run_originaldev.py`实现已预锁b05a019f公式，实际四个full64来源齐后才绑定plan。校验原B3必须等于同G/S/Q的固定二票，保全部B901与非关系对象；单完整预测freeze后最多1新v2。若字节等于B3、当前best或原901则继承指标，0重评分。编译及两项有区别投票合同通过，尚未实际装配/读gold。C full64仍待正式工程探针/训练，未新增质量结果。

完整 C 已到：64 fresh 全 strict、300纯关系，final-handoff78615a55；原临时服务/GPU已实际释放。已绑定[唯一原dev计划](plan-originaldev.json)，四源新增供给分别{'grounded': 28, 'sampled': 8, 'qwen3': 23, 'coded': 14}；当前best为05cfa90f/.769235669942。先推送源码/计划，再全预测冻结与最多1次新v2；保留全部901实体与关系，不做C单臂或子集扫描。
