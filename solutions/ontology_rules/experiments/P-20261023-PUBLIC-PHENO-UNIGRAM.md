# 单词候选供给与运行成本准备

目标是判断公开 PhenoTagger 的单词神经候选是否值得进入新的供给控制。此前单词遗漏只是诊断动机，本次枚举不使用那89个gold错误；使用固定train64四字段原文，全量、机械地执行现有tokenizer和原过滤。无originaldev，不拟合、不生成正式预测、不测试新质量。NIL目标问题不并入此项。

## 固定边界

原 `public_phenotagger_candidate_function.py` 中已有完整 `if ngram==1` 分支，但外层 `range(2,sentence_ngram+1)` 不会进入。此次只将外层替成 `range(1,2)`，独立输出sidecar；原2–8函数、旧候选与全部缓存保持原位。每个单词分别调用原 `pun_filter` 与 `pos_filter`，实际生成索引还逐项与独立过滤枚举核对。这里的单词是原NLTK token，不是任意空格片段或单个WordPiece。

缓存上游README只说明dictionary+DL混合；实际排除单词的作者意图未核实。不能称此项为原生默认或已确认bug修复。原`HP:None`也不能转成比赛NIL。

## 输入普查结果

| 项目 | 实际数量 |
|---|---:|
| 文献／段落／句子 | 64／2,301／8,697 |
| 原token坐标，全部恢复核验 | 205,656 |
| 标点过滤／POS过滤 | 28,351／55,892（交集0） |
| 原过滤后单词坐标 | 121,413 |
| 不同lower单词输入 | 13,145 |
| 原dict覆盖的保留坐标 | 1,985（涉及369种词串） |
| 保留坐标中无原dict跨度 | 119,428 |
| 原PubMed／Bioformer最终canonical已有这些跨度 | 2,010／2,015 |

两公开模型旧unique缓存实际逐文件SHA与全部行复核：PubMed249,116行（原16/48存在跨文件重复），Bioformer244,938行；全部native词串长度均为2–8，精确1-token tuple复用都为0。因此按既有缓存键策略，每模型各需13,145次新词串前向，不能使用另一个模型的logits。这里没有声称模型训练时未见该词、其子词未编码或任意旧多词串的张量一定不同。

两tokenizer复放全部13,145输入：PubMed词表30,522、`do_lower_case=True`；Bioformer32,768、`False`，但原表示层都会先`word.lower()`。原32长度、post-padding契约保持，全部输入无UNK，每模型实际输入ID序列也恰为13,145种。一个同样的长串被截断（原35／49 tokens含special），完整坐标仍保留；这是原32策略的限制，不针对gold删样。

程序42.591秒，外层48.373秒，低于120秒硬限；原token/坐标与真实tokenizer检查均完成，无模型框架或权重导入。回收后又逐行核全部121,413原文切片、词频和输入数组摘要，0.827秒。旧A100纯前向率仅给8.78秒PubMed／6.61秒Bioformer的参考，不含加载/网络，也不是A30实测。

## 环境与下一最小验证

实际A100原TF2.12/cu11.8/cuDNN8.6环境在CUDA隐藏下CPU import通过，2.611秒。其GPU2正用于序列训练，本项不占用。`.3`和`.100`的Python3.10 ABI与glibc满足私有packages条件性迁移，但没有在目标验证TF推断；原venv绝对解释器链接不能直接搬用。保守两模型全依赖迁移约6.89GB，未迁移；不能因目标存在CUDA12/13就认定CUDA11运行不兼容。

根线程决定不迁移，追加一次Bioformer单模型CPU吞吐probe：冻结13,145数组前256、两批128、原51,380,594参数/137变量native loader与H5，GPU实际设备必须空、TF inter/intra各2，90秒+TERM5硬限。只核真实输入、CPU前向、finite与耗时；不改变接受/旧pool。保守全量估计≤480秒才准备完整缓存计划，是否运行仍由根线程决定。实际probe已完成：17.754秒外层，导入2.967、加载6.299、trace1.220秒；两批3.0662/2.3729秒，全部finite、sum1、17266列，权重及输出均CPU、GPU设备空。保守估计11.4566+103×3.0662=327.273秒<480。准备独审d912、CPUprobe增量独审88ba均通过；未重复旧权重审计或NN。

根线程据此授权一次全量Bioformer CPU cache：480+TERM5秒、两线程，复用该256准确raw，只续12889词。固定原top1非None、native5dp严格>.95、既定current映射，输出新unigram sidecar，旧2–8/dict/原预测不动。失败保留已flush原始分数和progress，不完整不得进质量。

正式质量唯一基底另已明确为后继已审`selector-nested-control`四fold（+195实体/fullScore.727975656，f5482b85），不是原cut3 .725655389。保留该基底全部旧实体含nested及所有非entities对象；新项只用原否定+非交叠追加，与任何旧实体重叠/包含都拒绝，排序沿原score降序/length降序/offset升序/HPO字典序。全部4预测先共同freeze后一次v2、CPU≤60秒；预定保留为pooled两实体F1均值严格升且至少3/4fold该和不降。未通过即关闭，不换基底/阈值/词表。全量缓存已实际完成：239.276秒外层／238.288秒内，101批新前向221.986秒；全部13145键完整，复用256字节一致。固定策略留下529词／2516坐标（1933与原dict跨度重合、583不重合），其余12199词top1 None、415未过门、2映射无效。回收全SHA与0.675秒逐key/原观测/sidecar自核通过。无标签或质量读取；唯一质量入口已绑定完整终点，待执行。

## 恢复与证据

源码／计划：[public_phenotagger_unigram_preparation](../public_phenotagger_unigram_preparation/README.md)。结果与实际SHA索引：[准备证据](../evidence/round2/public-phenotagger-unigram-preparation.json)。外存 `ontology_rules/public-phenotagger-unigram-preparation/` 保存全坐标gzip2.94MB、全部词串/真实tokenID gzip331KB与真实state；Git仅源码/小索引/报告。运行环境独立证据在外存 `research/public-phenotagger-unigram-preparation/runtime/`，其小探针源码已收编。
