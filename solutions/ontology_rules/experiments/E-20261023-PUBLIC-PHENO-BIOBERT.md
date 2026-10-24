# 公开 BioBERT 第三个来源固定控制

目标是检验独立公开BioBERT表型head是否补充现有PubMed/Bioformer候选与来源信号。先完成全train64无标签缓存，再沿已保留四fold体系做唯一固定控制；名称不同或加载成功都不是收益证据。本次不读原dev，不改变已关闭的unigram/低门槛/top3/Sap/pair配置。

## 缓存：完整完成，无质量

固定原2–8gram生成器、原native五位舍入后top1严格>.95、字典/重叠/缩写和currentHPO映射。244938不同lower词串全部由独立BioBERT H5前向，不借另外两模型logits。原文CPU准备41.094秒：64篇/2301段/205656token坐标、296984候选和4130词典项全部核验。代码与plan在运行前提交10b6187。

A100 GPU2 fresh核空后23:48:23.549Z开跑，首个真实128批的词表28996/CLS101/SEP102/PAD0、32长度及128×17266输出finite/sum1通过；同一次全量继续，无弃置probe。纯NN157.241秒（1557.7词串/秒），native含准备/加载232.331秒<480，来源CPU4.123秒<120；stderr为空，managed已退出，23:52:15.981Z GPU2无PID/0MiB释放。

得到5903canonical键；原文坐标/旧→current合法映射/真实来源flags/64逐文行SHA全部自核。来源为neural3871、dict3949、abbr88，可重叠计数。首批与native完整manifest绑定；没有任务标签或质量评测。一次提前回收在未完成状态门退出，未创建archive、未改活跃源或重启任务，随后原作业完整收尾。完整模型/候选raw保留远端，必要小产物已回收，[源与恢复](../public_phenotagger_biobert_train64_cache/README.md)。CPU权重加载和一次TF环境修复另见[来源准备](P-20261023-PUBLIC-PHENO-BIOBERT.md)。

## 事前固定的唯一质量控制

原PubMed/Bioformer候选对象全部旧字段/类型/顺序保留，与本BioBERT native canonical键并集；只增加独立六个来源字段。每fold实际vectorizer维数不同，不强行添加不存在的列。新key按冻结原来源缺失约定，没有伪造Sap或raw后验；旧native hybrid分数明确不是类别概率。沿原四outer train48→16以及各三train32内层来源，不能挪作新实体槽的train32/F1隔离对照。

固定HGB160、cut .3和当前proper-nested解码，比较已保留selector-nested-control pooled Score .727975655663。所有非entities逐对象不变；重新fit可删除/替换旧预测实体，候选保持不等于预测保持。无标签preflight先核≤1.5倍行数与4fit/infer≤120秒预计成本；源/实际计划推送后执行。四个新完整预测共同freeze再唯一v2，主标准为pooled两实体F1均值严格增且至少3/4fold不降。没有cut扫描或额外基底臂，[授权政策](../public_phenotagger_biobert_selector/authorized-proposal.json)。

实际CPU准备10.870秒，所有244794旧对象projection的字段/类型/次序精确保留，仅加六个真实BioBERT字段；8池合计246804行（+2010）。每折真实train与held计数及成本依据见[preflight](../public_phenotagger_biobert_selector/preflight.json)。按历史实测最大56.999秒×最大行数倍率×1.25+10秒估算，低于120秒；这是预算估计，正式四fit仍有硬限。缓存增量独审1597f28f通过，未重复权重/NN。

## 训练侧唯一终点：达到预定保留标准

四fit/infer实际55.703秒<120，解码/冻结/评测2.973秒<30；16篇×4完整预测共同冻结cf8c1a28后才读新质量。实际vectorizer列数102/101/103/103，对旧96/95/97/97各增加六个来源字段。全部非entities和关系对象一致，0格式错警。

| 范围 | old menF1 | new menF1 | old docF1 | new docF1 | 两实体均值增量 |
|---|---:|---:|---:|---:|---:|
| F0 | 0.768070953 | 0.772285966 | 0.778625954 | 0.789674952 | +0.007632006 |
| F1 | 0.772658610 | 0.773782772 | 0.821126761 | 0.819649123 | -0.000176738 |
| F2 | 0.754549490 | 0.766990291 | 0.794470046 | 0.806953339 | +0.012462048 |
| F3 | 0.768878719 | 0.774487472 | 0.812962963 | 0.819012797 | +0.005829294 |
| pooled | 0.766299111 | 0.771948494 | 0.803367149 | 0.809789609 | +0.006035922 |

pooled Score **.727975655663→.730993616492（+.003017960829）**；mention净+48TP/+8FP，document+25TP/FP不变。micro .687846109006、macro .654390253486及1180TP/534FP/537FN关系计数完全相同。3/4fold实体均值提高，F1微降−.000176738；两实体pooled均提高，因此满足事前keep。此为已开发train64四fold结果，未达到用户原dev Score≥.8目标，也不是盲测或官方成绩。

完整模型/vectorizer、全候选概率和4份预测外存保留，模型真实SHA均在共同freeze引用中；独审已load实际模型复放，无重复拟合。四fold结案后获准的唯一原dev迁移见下方完整结案。缓存源与终点1597f28f已独审，选择器统一终审2907fc55已通过（全8池/四真实pickle所有held概率及预测逐字节、4+pooled v2/格式/冻结时序exact，无重复fit/NN）。[结果及恢复索引](../evidence/round2/public-phenotagger-biobert-results.json)。

### 固定原dev迁移（已完成）

四fold统一独审2907fc55通过后，原4×48 OOF来源完整64篇训练池61,781→62,281（+500）逐原spec核验，唯一HGB160模型103实际列/4,196正例，内部14.706秒、含进程15.084秒；模型42e94115在新dev推断前冻结。复用旧full64来源，公共BioBERT六字段同训练侧，dev只使用c0fc四字段输入。固定.3/proper-nested、当前d531模板全部非entities/368关系原样；先新完整预测冻结再唯一v2，旧指标复用，不扫阈值/重算关系支持。源码与[固定协议](../public_phenotagger_biobert_dev_transfer/fixed-dev-protocol.json)已保存；native同模型4,655缓存复用+50,548新forward，独立GPU上限180+TERM30，准备时尚无新dev质量，实际终点如下。当前完整实际库存9,597,467,990，已含一个共享使用的full64 LoRA；最初冻结协议的“shared weights/adapter另记”措辞需按文末澄清解释，原件不改。闭合实体/二元模型不计入生产链。

#### 原dev唯一端点（完成，统一独审PASS）

Score **0.737585917391→0.738493752650（+0.000907835259）**；mention +13TP/+17FP、doc +3TP/+1FP，mic/mac及368关系全部不变，0格式错警。1,379个native键并入旧13,144候选→13,246；同一个已冻full64 103列模型预测3.305秒（外3.524）。native78.392秒/纯新NN34.452秒，GPU已00:08:21.369131释放，来源CPU0.967秒；同BioBERT4,655复用全部原tuple/top3/native行/索引精确。

首装配1.572秒在读取gold或输出预测前因错误的metadata整对象等式退出：旧提交schema没有patient/full_text。只改为现有公共字段一致检查，原源/计划保留，独立attempt2在剩余28秒内用时1.669秒，未改任何数学/分数/输入。完整新预测SHA `baac8186…` 冻结 `a84d2ea8…` 后唯一v2；结果 `3f5d2aa0…`。合计3.241秒<原30，不追加臂/阈值；统一独审7336b280已通过，root负责更新受审best。

[结果与完整模型恢复索引](../evidence/round2/public-phenotagger-biobert-dev-results.json)保留真实pickle、全量候选/概率、预测与失败轨迹路径/SHA；大产物外置，不能只凭小summary重训练或重推NN。此为原dev开发正结果，尚未达0.8，不是官方分。

独立复核实际full64 pickle重放13,246个候选概率maxdiff0/JSONL字节一致，完整1,072实体预测（1,026 greedy＋46 proper-nested）逐字节一致；新v2/格式/先freeze后gold时序全部通过。原始raw仅复用真实来源与既有完整验证链，未重复NN/fit。备份82项合计1,847,310字节，含实际完整训练模型/vectorizer；后补独审包单列，四fold模型、大特征矩阵和rawlogits未混入。

参数口径澄清：当前旧333关系与新增35关系均实际使用相同full64 adapter `1a703b24…`，不是因目录含meta1而增加一个adapter；两原始manifest的base/total及adapter文件SHA完全一致。完整七项库存已含43,278,336 LoRA，总 **9,597,467,990**，剩 **402,532,010**；勿重复加adapter。[实际库存及来源说明](../evidence/round2/public-phenotagger-biobert-parameter-inventory.json)。该新说明只澄清措辞，不改变任何冻结预测、计划或备份清单。
