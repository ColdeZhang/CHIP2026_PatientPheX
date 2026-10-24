# NeuralCR 后继缓存与替换的静态准备

本阶段只读源码/既有元数据；未做新NN、训练或质量评估，未定全量GPU预算。完整CPU准备已独审，GPU成本探针仍待连接恢复。

## 缓存契约

冻结完整train64四字段输入和同一上游原生1–7词枚举；已有683438位置、408272实际token keys，512固定批序意味着797整批+末批208，共798批。缓存只保存这个模型的原始top1列、raw HPO和未四舍五入float32分数（另存bit pattern便于精确恢复）。查表返回的分数须恢复为上游同类型`np.float32`，并绑定实际NumPy版本；不能因JSON读取变为Python float后直接给原解码器，尤其阈值附近的scalar promotion语义可能变化。只在native解码后序列化为Python float。top1保持原get_match的argpartition选取，不暗换argmax的tie规则。所有key完整且唯一才宣布完成；禁止缺键补0/None、按FN筛选词串，或复用别的模型的概率。

首1536个probe输出若真正通过且完整SHA、模型/运行时/TF32策略一致，可在后续获准缓存时精确复用，避免重NN；目前probe尚无结果。原生get_match对不同batch形状可能有浮点差异，因此512批序是新缓存的固定定义，不宣称与逐段批量逐位相等。

将原生annotate_text方法保持原样，仅以严格查表get_match代替模型前向：候选和分块顺序不变，排除原root/None、严格>.8，原同ID包含抑制及按长度优先的重叠消解保持。NCR没有PhenoTagger的5位小数、字典混合或缩写传播，不套这些后处理。实际train64枚举没有空候选段；原生空输入边界仍保留，另处理需明确记录。

## 映射与证据

原生解码后再映射当前HPO，禁止先删head/重归一化后top1。沿已有公开Pheno映射政策重新构建NCR专属14260概念+None的完整列映射：同current有效ID→官方alt_id→单一official replaced_by，且落在2026-06-23 HP:0000118非obsolete分支；其余记录丢弃原因。旧17266列映射不能当NCR词表。完整onto.json目前在已验远端资产中，本地无副本，本阶段未虚报映射覆盖率。

输出local start/end以原文Python字符坐标核对，再加passage.offset；保存pmc_id、passage_index、原始/当前ID、raw概率及完整来源。映射前后分别计数；canonical重复键只合并相同(doc,offset,length,currentHPO)，保留raw来源。大缓存/全部预测/日志仓外，Git只存源、plan、小报告和SHA索引。

## 真正替换与隔离

必须从无BioBERT的冻结旧池重建：不能按biobert_present直接删除整个候选，因为它也标记与旧来源共享的键；应以原旧池键差集识别独有追加行，保留旧来源，去掉全部BioBERT六字段、模型加载/缓存依赖和旧103列estimator。NCR来源字段须按它真实纯NN行为重新定义，不能伪装成BioBERT或凭分数伪造字典/缩写flag。

每个新HGB/vectorizer都fresh拟合，沿原四fold pmc_id和所有监督派生资源核隔离。仅公共固定NN的无标签cache可跨fold共享；旧OOF分数不能因OOF名称就用于新下游划分（H40）。原dev16不参与当前准备；未来是否原dev迁移以独立正式方案决定。具体字段和代码依赖由feature-dependencies文档给出。

计入FT两矩阵的完整NCR491068890参数；只有实际移除BioBERT121587826及全部可执行依赖，原9597467990预算才投影为9966949054。不能保留旧模型后只改参数表。该静态准备未实施替换、fit或任何评分。

本地纯dtype边界例（NumPy2.3.5）验证同一float32(.8)标量与转Python float后`>.8`结果可不同；这是保持dtype的依据，不外推当前远端NumPy1.23.5，也未执行任何模型/质量检查。证据score-dtype-boundary.json。

## 关系间接依赖已闭环（01:57）

统一独审a09d7978核对冻结提示/来源及逐文原始association子串，确认e9→d531→baac关系完全原留；public-new只来自PubMed/Bioformer的42响应和固定pre-B6401支持保留35，a179新增23仅旧466概念池。因此可原字复用a179的association投影，无需重跑关系NN；不能复用baac/a179实体或旧103列selector。若改变固定支持政策才重算支持。完整review见relation-dependency-review.json；只有实际去掉BioBERT实体候选/6列/执行依赖并fresh HGB+解码后，9966949054参数投影才成立。本阶段仍是静态准备，未实施替换。
