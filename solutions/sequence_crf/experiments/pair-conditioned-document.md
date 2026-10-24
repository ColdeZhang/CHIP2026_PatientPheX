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
