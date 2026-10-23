# 第三个公开 PhenoTagger head：BioBERT CPU准备

固定公开BioBERT head已下载并通过原生CPU严格加载。实际201个变量、121,587,826参数全部与H5数组逐项相等；这里只证明权重及接口可用，尚无比赛数据预测、GPU吞吐或质量结果。历史源码/终点检索未找到已运行的BioBERT head控制；已运行PubMedBERT和Bioformer的结果不能替代这个版本。

## 来源与表示

权重是 `lingbionlp/PhenoTagger_v1.2@02ce3f529820dccca46bae168c1933ba25253125/biobert_PT_v1.2.h5`，实际486,624,416字节，SHA `1c4d799c667bef5ee0d7daef66ec30fca01f791bce91854566b10630815d7573`。准确ModelScope task仓库接口返回404；因此使用固定原始HF文件，不声称全站其他镜像都不存在。分块下载59.266秒，完整SHA一致后临时签名URL配置删除，未下载第二份基座。

沿已有ModelScope BioBERT配置和词表，之前及此次小文件复核均与固定HF副本逐字节相同：config SHA `a51b82d8…`，vocab SHA `eeaa9875…`。12层/768维、28,996词；该固定源本身没有额外tokenizer配置，不合成新策略。保留原表示层先将每词小写、再交给 `do_lower_case=False` tokenizer的实际流程，长度32、原padding/CLS/SEP契约。它不等同于保留输入大小写的BioBERT使用方式。

旧head保持17,266列，`HP:None`仍为17,265，原label顺序及固定旧→当前HPO映射全部核对；不把None转NIL，不改gold。公开PhenoTagger权重元数据为Apache-2.0、代码MIT；BioBERT基座镜像license字段较弱，保留原来源记录，不扩大授权结论。

## 真实CPU检查与一次恢复

复用A100现有私有TF2.12环境、原native loader，CUDA隐藏、实际GPU设备为空，CPU线程1。只从配置构建模型，再把全部浮点变量置NaN；原生topological `load_weights`，不使用 `by_name` 或 `skip_mismatch`。恢复后全201变量finite，形状和每个元素均与对应H5数据集相等，覆盖121,587,826参数。四个固定合成短语验证原表示、finite及17266列概率sum1，耗时.806秒；合成输出不是质量估计。

首次新启动环境漏带旧CPU作业必需的 `FORCE_TF_AVAILABLE=1`：TF已经import，但Transformers检测拒绝，5.375秒后在构建encoder前退出。该失败没有H5加载或前向，原source/state/stderr保留，不记为模型质量负结果。独立attempt2只补原标记，并把硬限降至174秒；checker数学和已下载权重不变。恢复成功外层20.054秒，程序16.391秒，含失败CPU总25.430秒低于原180秒名义预算。第二次stderr为空，无环境安装或6.89GB迁移。

## 若获准进入缓存阶段

原固定2–8gram train64有296,984坐标、244,938不同lower词串，旧dict为4,130坐标。这次不使用已关闭的unigram扩展。第三个模型需全部244,938次独立前向，已有另外两模型的logits不能复用；目前没有这个head的任务缓存。

参考同为BERT12×768的旧PubMed A100实测1425.5–1496.7词串/秒，纯NN类比约164–172秒，加100秒准备/后处理余量约272秒。这是资源估计，不是BioBERT实测速率；4个合成CPU短语也不足以推断完整CPU吞吐。任何下一GPU缓存需先按主线程调度/空卡检查和实际首batch预算执行。本准备阶段不启动GPU、fit、质量评测或原dev；名称不同本身不构成互补收益证据。

## 独立审查与恢复

增量独审 `f954be78` 通过：201库存/参数/列顺序、4×32实际表示独立WordPiece复放、固定下载来源和一次环境恢复均核验；逐数组相等与softmaxsum复用实际源断言，未重复H5/NN。审查通过不表示已获得质量提升。



[源码与计划](../public_phenotagger_biobert/README.md)、[实际产物索引](../evidence/round2/public-phenotagger-biobert-preparation.json)。外存 `research/public-phenotagger-biobert/` 保存原失败、恢复CPU和小证明，独立来源检查在相邻 `public-phenotagger-biobert-source-check/`；完整H5只保留A100项目自有 `public-phenotagger-biobert/`，恢复目录只用只读符号链接。下载和CPU均已结束，既有完整权重可复用，不重复下载或模型加载。
