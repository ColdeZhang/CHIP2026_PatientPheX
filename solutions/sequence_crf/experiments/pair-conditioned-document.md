# 全局患者—表型条件聚合：输入与 CPU 可行性

2026-10-24，原固定 train32=F2+F3 → F1；当前阶段只完成输入普查与独立数学合同，未读 F1 金标、未新增质量。旧 CE 拼窗、identity bridge、doc-HPO max-MIL 的负结果保留。本提案新增的是可学习的患者—表型查询对全文 token 的汇聚；不把“更多窗口”或 ATLOP 二分类阈值头称作新机制。

## 来源和原方法边界

ATLOP 官方提交 `1db77ab8bc4fe5f7f8c23067ea17055d58b310a3` 的 `model.py/get_hrt` 用患者/表型 mention attention 的逐项乘积、归一后加权上下文。若两端只在无共享 token 的独立窗口，该乘积严格为零。`long_seq.py` 仅首/尾两个 512 窗，不是任意长度全文实现；BERT 特例长 1500 时中间 478 个位置未编码。`prepro.py` 三个入口默认先截至 1022 个正文 token。不能原样承诺本任务全文覆盖。

固定 repo tree 和 LICENSE/LICENSE.md/LICENSE.txt 查询未找到许可文件，故只引用公开公式和小型外部来源证据；**不复制/执行原 ATLOP 实现，不使用 DocRED/CDR/GDA 标签或其权重**。本目录 NumPy 公式独立实现。

公开编码器复用已核 MIT 的 `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`，768/12 层/30522 词表/512 长度。公共六文件及现存 `.3`/`.100` 副本可恢复，不下载。分词器虽来自旧 CE 输出目录，但完整 vocab 与公共30522词表一致；已清除旧保存的 OnlySecond160 truncation 和 Fixed160 padding。没有加载旧 CE 权重。

## 严格输入与完整坐标

复用已审 H40 链 `F3→F2`、`F2→F3` 生成训练输入候选，F1 候选由 train32 生成；生成器、训练和输入校验均排 F1/F0。两组旧124训练提示含880概念/2506患者对，F1原80提示587概念/1557对，与此次全文重建逐对集合完全相同。未加入新候选、coref、训练词典或 HPO 名称输入。未知候选/缺患者 occurrence 硬拒绝，不默填全零。所有给定患者 mention、所有已有预测概念 occurrence 都保留；NIL按旧文字键，compound按旧原子键。

各 passage 按原顺序独立 WordPiece 后串成全文 token 序列，段间插入显式 `[SEP]`；每个原 token 保存 passage/local/global offset，插入分隔符 offset=None。编码时分隔符可见，汇聚时与 CLS/尾SEP/padding 一律排除。窗口510正文位置+2特殊token，**overlap128，推进382**（HF stride含义，不是每次推进128）。不限制窗口数；跨窗副本平均回同一原 token，先每 occurrence token 均值，再全部 occurrence 均值。重叠实体独立索引，不插入冲突的实体标记。

| 项 | train32 | F1 输入16 |
|---|---:|---:|
| 原正文 token | 113210 | 58522 |
| 旧 passage 独立窗 | 1127 | 574 |
| 新全文窗 | 307 | 158 |
| 新实际编码 token（含重复和特殊token） | 150114 | 77568 |
| 原/新固定pad512位置 | 577024 / 157184 | 293888 / 80896 |
| 原/新同顺序每文4窗动态pad位置 | 235407 / 155625 | 125455 / 80341 |
| 原/新动态pad注意力单头格数 | 61935705 / 79294159 | 36174459 / 41015845 |
| 患者/表型 occurrence | 255 / 1838 | 162 / 1189 |
| 全文打包后仍不共窗的候选对 | 1065 / 2506 | 602 / 1557 |

全部3444 occurrence 原文字符切片及完整落窗通过，最长13token；一项训练患者原边界处于 WordPiece 内，只池化与原跨度相交的token，不延长原坐标。48篇中47篇正文超过1022token，此计数是公共分词输入代理，不是运行ATLOP后得到的召回。旧passage分法有1616/1061对无共享局部attention支持；打包增加共窗机会，仍不能代替全局查询。

动态pad位置降33.9%/36.0%，但实际token增29.3%/28.9%，注意力格数反增28.0%/13.4%；**不能用少了约73%窗口宣称同倍率加速**。最长全文7729正文token、21窗；所有完整上下文 FP32 hidden 单文最高23,743,488B，32-pair块注意力最高989,312B，这些只是张量量级，不含编码器反传激活/优化器或峰显存。

## 唯一两臂与容量

对窗口编码并合并后的正文状态 `h_j∈R768`，患者/概念全部 occurrence 均值得到 `e_p,e_c`：

- `q=tanh([e_p;e_c]U+bq)`；`z_j=tanh(h_jW+bz)`，query/key/value宽128。
- 对照：`a_j=1/n`；新臂：`a_j=softmax(q·z_j/sqrt(128))`。
- 两臂相同 `c=Σa_j z_j`，`logit=Linear(tanh(Linear([q;c;q*c])))`，隐层128，BCE/fixed sigmoid .5。

两臂都是 **344577 个可学习头参数**，同初始化、同编码器、同输入、同训练预算。uniform 臂的 U/W 仍通过分类器参与训练，没有闲置一套查询参数冒充相同容量；注意力依赖关系改变是本次唯一设计变量。无需输出编码器各层 attention，也不构造全文 N×N 注意力矩阵。每轮train所有 pair×token=9,301,025标量，F1=6,107,289；实际计算/内存仍待真实实现测量。

这是全局 query head 的独立适配，**不是 ATLOP 原版复现**。全文token仍只在本地窗口经过BiomedBERT，上层仅一次全局汇聚；没有自动解决远距共指或输入患者身份冲突。均值端点可能稀释关键 mention。理论上完全相同 occurrence 集合的不同ID无法区分，实际本固定池该类组数为0；不因此加入新ID/name特征。

## CPU 合同与下一阶段边界

NumPy 13项合同通过：两臂两标签对全部98个小模型参数及输入状态做中心差分，最大误差低于1e-8；所有模块梯度非零，单次合成更新降合成loss；padding/分隔符NaN毒化完全排除；跨窗只有一端时局部乘积为0而全局context非零；改变患者改变条件上下文，uniform上下文不变；重叠mention、未知候选、缺端点、重复窗平均、等attention logits退化uniform通过。

这些只证明数学/接口，不是正式编码器autograd、BF16数值、吞吐或模型质量。原输入普查0.663秒、全文打包1.964秒、合同0.084秒、成本复算0.155秒；不新增质量评分。

根随后授权**仅真实短probe**：独立PyTorch实现、实际CPU合同后，fresh `.3` A30 GPU0，按输入最长train32文献、两臂各fresh同public encoder/seed/精度，encoder全部可训练并启gradient checkpointing；每次最多4窗、pairchunk32、encoder2e-5/head1e-4/clip1/wd.01，真实非零LR AdamW一步，180+15秒硬限，状态丢弃。先核旧CE class-loss是否适合文档；正式8ep是预算预案，**尚未获正式训练授权**，显存/吞吐测得后再决策。不关闭checkpoint以绕过环境不兼容，不把probe结果当质量。

## 恢复

外存 `R/research/pair-conditioned-document-feasibility/`：`plan/report` 原passage，`packed-plan/packed-report` 完整坐标/窗口，`head-plan/head-contracts` 合成数学，`budget-plan/budget-report` 同序padding复算；大token/offset/window文件保留外存。Git小索引见 `evidence/round2/pair-document-feasibility.json`，真实源 `pair_document_feasibility/`。所有源和输入SHA及实际命令在索引；无GPU/模型权重加载/任务训练/F1gold/originaldev读取。

实际Torch CPU 0.191秒通过，完整两臂四标签case的NumPy→autograd参数/输入梯度最大误差<1e-12；padding NaN、重叠梯度、未知候选、整文跨chunk weighted BCE与旧weighted CE等价通过。实际.3 torch2.7.0+cu128/TF4.50.0支持非reentrant checkpoint，六公共文件完整SHA无变化。最长**train32**是PMC5793168/7023正文/19窗/5对，连既有训练标签后5对全负，未换样本；全局2506对594正/1912负，权重w0=.655334728/w1=2.109427609，整文sum(wBCE)/sum(w)，不逐chunk改分母。旧CE按随机16行batch与本次docbatch不同，但双新臂完全相同。工具首次绝对路径及随后大payload失败都发生在CPU/GPU之前，已留attempt并改mount相对路径/256KiB分块；数学/输入未变。新probe-plan已冻结，真实GPU尚未开始，索引pair-document-probe-plan.json。

实际双fresh probe已完整PASS并释放：managed `proc_b9270cc8` 00:57:15.239Z启动，00:57:26.662结果完成，含import约11.423秒；函数内部计时7.692秒不含import。uniform/pair训练步1.822/1.206秒（后者环境已暖，不作速度结论），峰2.09GiB；实际108891648 encoder+344577头、全部encoder梯度finite，嵌入/首末层/U/W/A/o均非零且参数改变。原生源码和完整公共权重SHA闭合，未保存probe状态。00:57:38.068GPU核空；managed仅running=false、无独立退出码，不写未观测rc0。后续8ep两臂共4912窗口/2401824编码token，按窗粗代理391–471秒，逐文全当最长约933秒；最大pair文372对/12窗未单测，不能把5对最长文当head上界。根同意准备共享1800+30正式方案，仍未启动；完整结果见pair-document-probe-results.json。


## 正式匹配双臂：8轮协议（准备完成，待统一入口审查）

根于01:01明确授权本协议在增量入口核验/源码推送后启动，无需重跑probe。训练文献32、2506对/594正；F1输入16篇/1557对另存gzip，严格无label/answer/association。相同public encoder、同seed与实际初始化hash、相同8轮文献ID顺序、每文一个AdamW步、每臂256更新；encoder/head常数LR2e-5/1e-4、wd.01、clip1、FP32/eager、非reentrant checkpoint、窗micro4/pairchunk32、整文weightedBCE分母全部一致。只有uniform与pair attention不同。原包装入口误导入无需使用的tokenizers依赖，两次在写文件前失败；改为纯JSON四helper，未重分词、输入与模型政策不变。

共享GPU总预算1800s+TERM30，含两臂fresh加载/8轮/final checkpoint/末轮F1推断，按实测最长文仅作保守成本代理，不能保证完成。每臂只存末轮model/optimizer/RNG及小进度；中断只保存明确不供推断的partial，不把部分轮次当最终模型。两臂完整1557概率/全16预测共同冻结后，单次CPU端点≤30s才读F1gold，fixed.5，无阈值扫描、F1早停、原dev访问。旧同池SFT32指标.660942003、规则.644761594复用已审metrics（后者不是F0 .711353522）。新pair仅在Score同时胜uniform/SFT/rules且macro对三者均不降时保留；两臂微正但未胜强基线只能作机制证据。

源码与恢复：外部 `formal-v1/formal-plan.json` b26d6df5、`endpoint-plan.json` a455dcbe，Git小索引 `evidence/round2/pair-document-formal-preparation.json`；训练和F1数据分别364,776B/195,670B留外存。部署仅上传冻结源码/输入并CPU编译，启动入口强制fresh GPU/无已有模型目录/统一审查PASS/已push commit；managed job另记 `job.json`，恢复只读原job，不盲重启。最终模型仍未训练、尚无质量。


正式入口独审 `e5180c9b…` 通过，源码检查点 `c9043ad` 已推并核远端；01:07:33.525983Z fresh核 `.3` A30 GPU0 无compute PID后实际启动 `proc_e34bff71-98fe-44d4-983b-698751e431d7` / PID1164449。外层共享1800s+TERM30，主截止01:37:33.526Z/最晚01:38:03.526Z。远端仅 `sequence-round2/pair-document-formal-v1`，以原job读取输出恢复，不能重启；本地 `formal_operations.py collect --directory .../formal-v1` 运行中只报告状态，结束后才回收小文件/确认卡空。当前训练中，无验证结果。
