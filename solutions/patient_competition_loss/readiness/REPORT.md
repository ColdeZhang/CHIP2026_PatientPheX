# 同一PCL模型跨折校准：只读可行性

核对象是已结束PCL末轮checkpoint **9ad43c5e…**（24/768、32篇F2F3），不是重新训练的模型。未拟合阈值、未读取新gold、未新建预测/Score/NN任务。

1. **现有F1 raw足够支持受限的纯CPU重解码。** 1542个NER≥−2物理跨度保存全部4285 patient logits；其中1343个NER>0。408个NER>0且至少一患者>0的跨度已规范化，逐条分数与patient IDs均和raw exact；另935个全部patient≤0未规范化。固定globalNER>0及严格patient>t、t≥0时，后者永不入选，所以无需再跑模型或规范化。**准确完整性限于max>0的关系及t≥0解码；非正关系的全候选真实max不完整，不能填造−inf冒称真值。** 当前没有显式relation-max表，但可沿已保存的accepted identifiers/原patient IDs/物理跨度，按(doc,pid,literal)取max并保留全部来源及ties。阈值与max可交换。此结论不覆盖t<0、改变NER门或将>改成≥。

2. **F0纯输入新推断可以保持训练资产排除F0/F1。** 实核NN train32与两纯输入文献ID交集都为0；StrictNormalizer的监督aliases精确也是38c353ea同train32。HPO/Sap encoder/public bank均公共，无B0/CID/selector依赖。准确说法是“train32 strict aliases＋公共Sap”，不是全公共规范化。F1已经用于此前模型/方法开发，因此未来即使阈值只在F0选择，也应称新固定开发验证，不能称首次盲测。

待授权后才需要实施：用原prepare.geometry(training=False,max_span=32)构造F0全部16/47的原text/offsets/锚点/候选/510×384完整窗口，并核原生tokenizer和全部coverage；原prepare.main会重建训练且写死F1/39，不能直接调用。建立held-only入口严格load206模型state、eval/no_grad全量输出；原train.main会训练，不能调用。normalize仅机械适配16/47与新status检查，strict/Sap policy、训练alias、输入跨度门不变。关系源不需要B0/CID，实体框架可为空；不在此阶段构造任何有监督F0候选。

成本依据：现存DocRE metadata中的原未加marker token长度为61460/max7098；与F0纯输入全文逐字相同，按510/384有164窗/max19，最多9患者。此处仅用来估算，**没有复用DocRE候选、marker或编码作为PCS输入**；真实F0 PCS几何仍需CPU构造核验。已测PCL F1推断循环1.232秒，固定norm子进程10.096秒；旧48篇CPU几何30.262秒、连原生合同44.052秒。估计F0 CPU15–30秒，模型加载＋完整推断＋norm20–45秒；实际query数和几何尚未测。若root后续安排，可考虑单一120+TERM30推断窗口；本报告没有启动或预占GPU。

准确输入、源码、raw及祖先refs与计数在result.json；check.py只读现成raw/selected evidence、metadata和两纯输入，不调用model/normalizer/evaluator。原两质量/负结论不变。
