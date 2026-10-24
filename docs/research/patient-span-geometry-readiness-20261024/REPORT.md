# 患者条件跨度：显式几何特征的有界只读研究

结论：当前PCS头**没有显式患者相对距离、同段、同句或病例归属特征**。现有train32提供了可区分的锚点几何；在原24轮时长控制完成后，值得考虑唯一的低容量几何残差对照。它不是首次使用位置特征，也不是最近患者规则；新点仅是将全部锚点的可学习几何直接用于现有患者跨度logit，并继续以概念袋监督选择潜在实例。本研究未修改owner代码、未fit/NN/GPU、未读取held gold、未评分。

## 1. 当前源码到底用了什么

以实际已执行 `patient-conditioned-spans-v1/collected/formal-source-v1/model.py` 与 `prepare.py` 为准；24轮正在另行准备/执行，不改它。

`encode_document` 对完整重叠窗口隐藏状态可微平均，再对每个患者的每个锚点内token平均、多个锚点平均，得到p。`patient_scores(hidden, patients, pairs)`只有三个张量输入，计算：

`score(p,i,j) = <R_i(q_i+W_p p), R_j(k_j+V_p p)>/sqrt(d) + b(p)`。

其显式RoPE相位相乘为 `R_i^T R_j = R_(j-i)`，编码的是跨度两端相对位置；患者锚点的坐标a/b没有进入这一式。患者p及token隐状态仍可隐式携带上下文/窗口位置，不能称为完全无位置信息；但不能据此称已输入患者到跨度的全局距离。窗口BERT的位置在各窗重新从头计，跨度RoPE没有补出锚点到跨度的坐标。

`prepare.geometry`保存`patients[].spans`、`anchor_owner`、`coordinates`的原段index和全局token/字符offset；这些几何目前主要用于定位、去重和标签/袋映射。`patient_scores`没有读取`coordinates`、段号或sentence ID。原段的type/section_type没有带入编码后的passages；不能将同段当“该病例归属”，也没有人工病例段落标签。

训练现有关系目标是 `BCE(max_{实例∈概念袋} patient_logit, bag_label)`，813正袋/2381负袋，1无可表示正证据袋masked。全局NER仍为GP multilabel categorical loss。`cpu-v2/preparation.json`残留旧balanced-BCE说明，**实际formal使用GP**；v2的train payload与实际GPU输入decoded SHA `3878fa3c…`完全相同，可用于几何研究，不可据该旧metadata改写实际loss。

## 2. H59查重：哪些已做，哪些没有

| 真实既有路线 | 实际使用的结构/监督 | 与本次唯一建议的区别 |
|---|---|---|
| CRF关系LR / `sequence_crf/run.py:pair_features,fit_relation` | 已学log_dist/log_prev_dist、同段/同句、nearest、竞争margin、section及病例词；每个概念匹配实体均继承患者概念标签 | **可学习几何本身早已做过**；它把袋标签展开到全部实体，非现有latent max袋，不联合患者跨度头 |
| 病例检索 / `case_retrieval/run.py:patient_pair` | 距离桶、同段/同句/最近患者/病例section等加入检索特征 | 不等于无结构基线，也不是joint神经跨度+latent袋监督 |
| identity-bridge真实BF16双臂 | 按同段/相邻同section选原文锚点，追加TARGET/OTHER及数字distance文本，CE按patient-concept训练；F0 Score .652285→.641438负 | 已给模型结构信息，不能称首次身份桥；不是全部锚点的数值特征、不是span-bag head；保原负结果 |
| 全文pair-query双臂 | patient/concept多提及均值→query→全局条件attention；pair .557825、uniform .542016，均弱于旧.660942 | 没有显式相对锚点坐标；学习单位是概念对而非全部可提议跨度 |
| DocRE/marker/30ep | 多提及池化、局部attention交集、双线性分类；三个默认cut控制负 | marker编码与attention局部性不是数值患者几何；没有当前NER+latent span袋联合目标 |
| 旧max-MIL背景过滤 | 256token概念局部片段、patient/phenotype markers、CLS头、所有occurrence max，3ep固定.5；.711354→.702753负 | 确实已有max-MIL，不能再称新的；没有全部患者锚点显式几何、完整共享文档跨度头；仅删旧关系 |

在这些指定机制与真实终点中，未发现已经完成“全部患者锚点的显式可学习几何残差 + 当前joint NER/latent span bag”同机制实验。该结论是有边界的实现查重，**不保证所有未知历史版本中绝无相似方法**，更不把旧负结果隐去。

## 3. 只读现成train32几何的实际供给

本次 `census.py` 仅打开已有`cpu-v2/train.json.gz`，解压115,076,256B、SHA `3878fa3ca531574be0f8eae6d5ec8f17f116bd0249ca484d1ea66666b0e705c5`；4.684秒完成。未重新tokenize、未打开held编码/金标、未用新预测。

- 32篇、87患者，其中17篇多患者；255个物理锚点，跨患者同坐标锚点0、完全相同锚点集合的患者对0。
- 当前训练关系头实际用2311个unique袋实例跨度，对全部患者形成6612次patient-span打分；全部锚点组合18,775个。全局NER的3,197,738候选不变。
- 多患者文献的1503个实例中，1421个（94.54%）的所提几何直方图能区分至少两个患者；82个对全部患者几何相同。它衡量表示区分度，不是正确患者识别率。
- 正袋中720/813（88.56%）至少一实例与目标锚点同原段；负袋357/2381（14.99%）。仍有93个正袋没有同段实例，因此同段不能作为删除门。
- 简单标点分段的同“句”供给：正212/813、负22/2381。只是原段内确定性标点片段，不是人工句子/病例归属；此次建议不使用它，避免同时引入额外边界假设。
- 560个多患者正袋中，421至少一个实例的目标患者为最近者；另139个袋每个实例都有其他患者严格更近。**139不是错标、证据错配或模型错误计数**；正袋只保证概念关联，不保证哪个跨度是证据。
- 原旧census的170是814弱最近目标中另一患者更近，包含不同NO/可表示性口径。本次不把170放到813/560分母，也不把它当人工归属真值。

完整数量/定义在`train-geometry-census.json`。所有token距离使用现有同一全文token坐标；同段用原passage index，窗口重复不增加锚点。

## 4. 成熟方法依据及迁移边界

[Position-aware Attention and Supervised Data Improve Slot Filling](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf)（Zhang等，EMNLP2017）的作者[TACRED代码](https://github.com/yuhaozhang/tacred-relation/tree/18221efc28be400ad3ec3939f1bbd32ab74b64f5)明确将subject/object相对位置送入可学习`pe_emb`，再作为attention附加特征；固定commit18221efc，Apache2.0。

[DocRED](https://arxiv.org/abs/1906.06127v3)（Yao等，ACL2019）的[官方多实体文档基线](https://github.com/thunlp/DocRED/blob/64622e608d06e83deda6b6e0e871effb9d1fd74c/code/models/BiLSTM.py)用`h_mapping/t_mapping`聚合实体，再拼接双向`dis_embed`送双线性头；固定commit64622e60，MIT。这证明显式实体距离与语义池化是独立信息源，而非让encoder必然自行学会。

两者均不是本项目的多锚点latent span bag复现。尤其DocRED基线距离由其上游实体对定义，不可冒称已实现本建议的全锚点平均。此次只下载少量公开代码/README/LICENSE，没有下载这些标注数据或权重；不引入外部人工表型标注。MRC/患者query条件化本身也不能自动提供缺失的原文锚点坐标，不能用“问了哪个患者”替代实际位置特征接口。

## 5. 唯一建议：50参数的全锚点几何残差

**值得一个后继单变量控制，先保持当前24轮对照不动。** 若root后续安排，使用与该24轮完全相同的fresh基座、顺序、优化器、学习率schedule、候选、全局NER、max袋/权重、归一化、cut和两个端点，只增加下式及其预先固定的独立学习率组（属于同一个50参数机制，不扫描）：

`new_score(p,i,j) = old_score(p,i,j) + mean_{a∈p的全部唯一锚点} E[bucket(signed_gap([i,j],a)), same_passage([i,j],a)]`。

跨度[i,j]含两端、锚点[a,b)不含右端：锚点在前用d=i-b，在后用d=j+1-a，重叠或紧邻为0。距离桶为0或`sign(d)*min(12,1+floor(log2(abs(d))))`，取值−12…12，实际表下标统一加12，共25桶；同段2类，共**50个可学习标量**。E用`nn.Parameter(torch.zeros(25,2))`在既有头初始化完成后注册，不创建默认随机初始化Embedding、不消耗任何RNG；保持旧参数的初始化与初始logit一致。原encoder/heads仍3e-5，新增E独立参数组固定LR=1e-2，weight_decay同.01；两组同原24轮768updates/76warmup线性衰减。此单一新组比例不是从held分数搜索。正负袋BCE经原max反传学习E，不把任一跨度硬赋给患者，不删除远处实例、不额外造证据标签、不依held扫桶/权重/cut。不引入病例段落归属或同句规则。

学习尺度理由：零初始化50标量若也用3e-5，768次实际使用的scheduler因子之和384，LR面积仅.01152；新组1e-2的面积3.84。此为“每步Adam归一化方向恒为1、忽略wd/瞬态”的尺度示意，不是实际更新量或严格上界；避免把几乎不能移动的残差误判为几何无效。已做纯Python schedule求和，0模型/0优化器/0fit，结果见schedule-arithmetic.json。固定较大新头LR是此几何控制的组成部分，不能把结果解释成纯输入特征、完全相同优化器的严格消融。

具体插入点：`prepare.geometry`缓存每token/患者锚点的原段index（现有offset/passages即可派生，不用新标注）；`PatientSpanModel.__init__`在原初始化后加零Parameter；训练器保留旧参数迭代顺序形成原3e-5组，并追加仅含geometry表的1e-2组；`patient_scores`增加几何参数，在原score返回前加残差；`loss`的`pairs[used]`与`infer`的selected chunk两处传同一几何接口。encoder、NER头、baglogits max及decode不变。不要给训练缓存全部P×3.2M候选几何，也不要另跑encoder。

成本：训练一轮新增18,775次表查询/平均，较现有6612 patient-span分数很小；最大现成train33锚点，在现有8192 inference chunk最多270,336 anchor-span索引（int64约2.16MB/单张量；实际总峰须工程确认）。50参数及Adam状态可忽略于108M encoder，但不虚报测得GPU加速/零开销。约60–100行几何/头接线及几项合同即可独立实现，无新模型下载。

必要合同建议（本研究未运行微型模型）：加Parameter前后RNG状态完全相同、同seed旧encoder/heads逐值相同、零初始化复现旧score/loss及旧梯度、optimizer旧组参数顺序不变/仅新50参数进入新组、两个组首步/峰值/尾步LR按冻结式一致、改锚点坐标而hidden固定时仅新残差改变、锚点排列不变性/物理重复不重复计数、全部实例仍入max及远距袋保留、训练和推断相同桶映射。主终点沿原固定B保护并集Score并完整报告macro；对当前相同时长无几何臂配对，两个输出先共同冻结，禁止用训练几何相关性冒称held收益。无明确增益则关闭该残差，不能顺势扫阈值或多组位置设计。

当前状态：研究建议，**未创建新GPU计划或启动实验**。源码/统计/成熟来源的精确指针见`result.json`。
