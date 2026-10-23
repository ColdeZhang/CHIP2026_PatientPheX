# 公开 PhenoTagger 新来源的最小选择器控制

2026-10-23，E-20261023-009。仅查已有源码、manifest和已完成的聚合报告；没有新训练、推断、评分或原dev标签读取。11份实际来源另存不可变快照，12套旧配置的候选/CRF来源哈希检查通过。完整元数据报告SHA `29b29ecb706a5666b6c0a237174e2886bc2111c7aecd1980f69f40c6d124d9dd`，见[证据](../evidence/round2/selector-readiness.json)。

**结论：可学习的全池选择和冲突替换早已严格做过；目前缺的是新来源覆盖全训练文献的可靠缓存，不是再发明一个分类器。最小新控制应为原HGB选择器加公开PhenoTagger候选，暂不为凑“全模型融合”再训GLiNER或CE。**

## 已有能力与真正缺口

`ontology_rules/calibrate_multi.py` 已用三套train32→inner16候选训练outer48选择器，后来四个outer折都完成。它先汇合本体、Sap和CRF exact `(offset,length,HPO)` 三元组，再对**整个池**预测概率、按`(-p,-length,offset)`贪心选不重叠跨度；不锁定任何旧实体，旧实体可删除、边界/ID可被高分冲突候选替换。因此“从只补非重叠改成允许删换”不是全项目未做过的机制，只是后续GLiNER/自训Pheno固定追加管线没有使用该能力。

但这份已审完整池**不含GLiNER和公开PhenoTagger**。旧三内折spec只有本体/Sap/CRF及可选CRF置信度；不能将已有tree称作CRF+GLiNER+Sap全池实验。现有GLiNER同协议训练32缓存只覆盖所选inner16等个别方向，未具备在outer48三组训练文章上均不见自身的完整同模型OOF特征。用已训outer48模型预测其训练48并喂入二层，会让二层看到过拟合分布，不能冒充严格OOF。

| 信息 | 已有状态 | 新来源所需补充 |
|---|---|---|
| source flags / exact / base / source一致跨度 | 已有本体、semantic、CRF及CRF-span标志 | 新增publicPheno标志、明确hybrid/neural/dictionary来源；原重合项合并，不制造重复标签 |
| CRF span/link置信 | 已从真实各折checkpoint导出，键与冻结输出exact；已做控制 | 不重训或重复导出；不能从仅最终entities杜撰置信度 |
| Sap概率与top1 cosine | 各折已存在，标签/别名银行依赖该折训练集 | 原同键可复用；新Pheno-only键若没有Sap记录须显式缺失，不能假装获得本折分数 |
| Sap margin | 旧特征未用，旧scored缓存通常不存top2 | 不在最小控制中补GPU编码或新margin变量 |
| 长度、词数、否定、章节、邻词、重叠和文档内支持 | 已有约84列，语境为固定小字母表 | 可直接从每篇给定输入构造，不用验证标签 |
| GLiNER score | 已有局部单模型缓存，但没有完整三内折同协议OOF | 最小控制暂不加入，避免额外三模型训练和来源混杂 |
| publicPheno score / top3 | 新native入口保留每词组top3真实概率与5位原格式、字典原候选和hybrid各阶段 | 全train64输入缓存未齐；需最终manifest、当前HPO映射和每实体来源对齐。最终hybrid score可先用，暂不另加top3 margin |

旧结果不应重复：四折pooled无CRF为.685837901、source为.692527322、confidence为.692974449；按事先“差小于.001选简单”选择source。单fold0置信树.714807668不代表四折全选该分支。另严格三小折CE新列控制（6个低层候选池→3个CE模型）耗289.18秒，.714807668→.701372111，已负结案；本提案不重跑CE。

## 公开推断可以一次运行64篇，但监督资源不能一起共享

固定公开 `lingbionlp/PhenoTagger_v1.2@02ce3f529820dccca46bae168c1933ba25253125`、权重`d58af55a…`未用本赛事标签。当前native入口只读无entities/association输入，固定公共2023字典和旧head原列序；输出后映射指定2026 HPO，不加祖先。每篇缩写和词组去重均只依赖输入。因此按**相同已冻公开模型、分词/阈值/词典/映射协议**运行一次train64无标签文本，按文献切出所有内外折，可以复用这层公开原始分数和候选；已有nested16属于train64的一部分，也可按逐篇输入/来源SHA复用，补剩48，不必四折重复公共前向。

这不表示当前脚本能原样换路径：`run_native_entities.py`硬编码16篇及候选计数，扩64需要独立参数化入口/manifest与逐文献SHA绑定，保留模型数学和解码政策；不得静默去掉原检查。还要保留raw neural/dictionary/pre-post abbreviation，不能只凭最终hybrid score猜原source。

仍须按折构造/选择：赛事train别名、CRF权重与置信度、Sap本体+本折别名bank及监督phenotypeness LR、任务微调GLiNER、任务CE/Qwen特征，以及最终HGB训练标签。纯公共Sap encoder和纯HPO embeddings可共享；alias扩充、拟合参数、top1检索结果随bank改变，不能共享一个全64 Sap结果评内折。监督二层的任何进一步OOF也必须让低层生成器排除未来二层heldout，不能只检查直接预测文献。

## 唯一建议控制（待新来源终点成功后，尚未执行）

复用已审outer0三套train32 OOF、固定source-HGB160轮/叶15/minleaf35/l2=5/lr=.06/threshold=.4/early_stopping=False。对照重放既有source池；实验只把同一公开native政策的canonical候选合入全部三内折及outer0输入，添加`public_present`和真实`public_hybrid_score`及来源可用标志。原池字段保持原样；public-only且没有旧Sap键的记录用**显式缺失标志**配合原CRF-only的占位约定，不能把占位.5/0记成实际Sap概率/相似度。既有source列和上下文特征复用，不增加CE/GLiNER新训练、margin或阈值变量。

两臂均由完整树选择、允许删换冲突实体，再用相同患者规则；此处不用当前“保全部CRF+只追加”的强基底冒充匹配对照。两fullpred冻结后仅一次outer0 v2，完整报告删/增实体及关系TPFP。若不高于匹配source-HGB、或来源不能精确回溯则关闭，不按该fold错例再加特征。若有正信号由root决定后续，不自行开原dev。

已有每折43,897–45,113候选的source-HGB全CPU耗10.22–12.39秒；所以新公共缓存齐后两树是秒到分钟级成本，建议CPU120秒cap，不预承诺新增池规模下耗时。**主要新增成本是公开64篇前向**：当前仅16篇65,406唯一词组，冷首128批投影946.53秒，尚非实测稳定吞吐；不能直接把64篇称几秒任务，也不能无文长计数就把4倍估计当预算。先等正在跑的native16实际终点/互补信号，再决定是否值得补剩48缓存。本次仅给可复用入口和来源判断，未启动该控制。

后续分工更新：root已安排独立helper沿不变public模型/decoder补无标注train64缓存（复用既有16篇，其余48，另有600秒cap）。本路线只继续固定HGB±public来源的CPU接口/计划；缓存未齐不启动拟合或评分。旧base若实际源码、输入和配置哈希全一致，直接复用已独审预测，不为多报一个对照重复训练。
