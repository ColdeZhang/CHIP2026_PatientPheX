# 公开 PubMedH5：train32 CPU准备与单一控制提案

目标是验证公开成品分类器能否借官方训练折的正例及可解释负例改善完整Task1；不是继续旧1749词面回放。仅公开PubMedBERT原H5 `d58af55a`、原17266列/HP:None=17265，训练文献严格F2+F3共32篇，held F1原16完整保留。当前没有fit、真实模型权重加载、GPU或新评分；原dev和heldF1 gold没有读取。

## 实际数据合同

所有正例从这32篇标注出发，**不要求曾被旧模型识别**。在包含标注的完整原文段上重用原句分/POS、原修正ngram生成与字符还原，得到阈值前真实query跨度可达性；包含原POS/punctuation过滤，不用粗2–8词数代替。这里只证明查询生成可达，不证明后续神经阈值、dictionary或重叠合并保留。

2197条可映射正例记录中，1481在原native查询集合，712不在原2–8/POS候选集合，4未精确对齐token。另明确记录44 NIL、11原head缺列、15多canonical复合ID、8非有效HPO分支；这些不伪造为None，并保护同实际输入免受负标签污染。负例来源另取固定公开PubMed已接受的神经候选cache，task-training=false；没有使用含heldF1监督来源的train48混合候选池。原始87 NO、451边界误报、277非实体误报分开，纯字典/缩写979个span不作为神经负例。

原实际Transformers4.30.1/tokenizers0.13.3/NLTK3.8.1/TF2.12 CPU编码为1658种3×32输入：**0截断，未截断最长21token；不同lower-query映射碰撞0**。同实际输入95组正负冲突、2组不同正canonical冲突、71组unsupported-positive保护、238组无native候选 occurrence被排除。最后1252种唯一训练输入：811正例、27显式NO、237边界负、177普通候选负。每query权重1，不重复曝光；负例预设hash配额未触发。31个正输入对应多个等价原分类列。

## 稳定目标与小CPU证明

单canonical目标映射到所有等价原列；真正多canonical compound暂不监督，但完整held不删除。目标为 `logsumexp(all logits) - logsumexp(target logits)`，不用softmax后clip。由原图最后Dense.input取得hidden，乘**同一个**Dense.kernel并加同bias取得预激活；原推断Dense/softmax与权重结构保持。

一次TF2.12 CPU合成证明0.329863秒：单列退化、多个等价列、None、极端错类梯度5组通过，空/非法/重复列5组拒绝；原Dense同变量预激活重现softmax，单列loss差0，最大梯度差1.46e−11。这里只证明小图数学/接口，不证明实际PubMed全encoder反向显存或训练质量。[数值结果](loss-result.json)。

## 预定控制与成功标准

建议单轮fresh原生TF Adam5e−6/batch64、全参数float32、不早停，共20更新步，不增加多配方扫描。严格原H5逐变量加载后才训练，probe状态不复用作正式模型；具体正式GPU预算尚未批准。

**Primary**：只替换PubMed贡献，其他两public/base/Sap/CRF、103列向量器、固定F1 LGB `a465e716`、cut .3与decoder全保持；完整16预测原关系及非实体字节不变。相对既有完整强Task1 `8cb61ba9` / Score .7286975364690524 严格涨Score才保留，全部分项报告。新分数分布对冻结LGB的迁移是此完整控制的一部分；若负只关闭此配方，不泛化为所有公开权重适配无效。

**同源诊断**：沿既有原PubMed fixed overlay协议（旧完整`24d18da2` / .6583570664643756），只换native输出，原base `8acd`及assembler/negation/overlap政策不动。与其自身旧终点比较，不把这个较弱Score与primary绝对值混比，也不在看到质量后把诊断改成主臂。两新完整预测先共同freeze后才读heldgold，各唯一v2；旧指标复用。当前只冻结方案，未生成新预测或调用评测。[完整比较](proposed-control.json)、[同源精确附录](comparison-addendum.json)。F0复验和原dev迁移由root之后另定。

## 成本与运行证据

本次native CPU14.164822秒；含两个独立Python启动的整个作业18.536317秒rc0，GPU可见为空，A100现有patient GPU作业未受本任务调度。14件终态archive188092B逐SHA重开核验。主要request/编码query/occurrence审核留仓库外且在A100私有持久目录，Git只留源码与小清单。[完成清单](completion.json)。

同架构既有F1 native65406 unique查询forward实际45.881796秒、完整75.315122秒；另48篇新增NN119.951556秒。它们都不是反向/Adam实耗，不能据此冒定正式训练预算。若root后续授权，需要一个短工程probe实测全encoder反向、首/稳态更新及约491MB H5保存/重开，再按20步加加载/保存与余量冻结预算；现在没有启动该probe。原数据阶段及数值合同没有改变官方B、head probe或既有全文formal。

前置两次metadata核对修正（原ml_ner路径、两源码仅CRLF/LF差异）在模型/数据CPU作业启动前结束，原记录保留；只有一次实际完整CPU作业，没有训练重试。
