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

根线程决定不迁移，追加一次Bioformer单模型CPU吞吐probe：冻结13,145数组前256、两批128、原51,380,594参数/137变量native loader与H5，GPU实际设备必须空、TF inter/intra各2，90秒+TERM5硬限。只核真实输入、CPU前向、finite与耗时；不改变接受/旧pool。保守全量估计≤480秒才准备完整缓存计划，是否运行仍由根线程决定。probe结果另补本记录，不把它当质量实验。

## 恢复与证据

源码／计划：[public_phenotagger_unigram_preparation](../public_phenotagger_unigram_preparation/README.md)。结果与实际SHA索引：[准备证据](../evidence/round2/public-phenotagger-unigram-preparation.json)。外存 `ontology_rules/public-phenotagger-unigram-preparation/` 保存全坐标gzip2.94MB、全部词串/真实tokenID gzip331KB与真实state；Git仅源码/小索引/报告。运行环境独立证据在外存 `research/public-phenotagger-unigram-preparation/runtime/`，其小探针源码已收编。
