# 唯一候选：原始 LongT5-TGlobal-base 患者集合生成

建议进入一次有界工程探针的候选是 **`google/long-t5-tglobal-base`**，不是再开一条Qwen目标/loss变体。它把完整报告与全部患者锚点放在双向长上下文encoder中，短decoder通过cross-attention产生目标患者的整个关系集合；当前Qwen是单个因果decoder按提示前缀自回归。既有DocRE/GP是判别式候选或跨度模型，SCST/DPO改变目标与优化，teacher/quote补全改变监督供给，都不等价于这一架构变化。

静态检查后root已授权单base资产和600秒有界probe，正式训练仍未授权；当前实际状态只在WORKLOG维护。通用C4没有医学/指令专门先验，87样本少；**唯一目标改为原强G的完整87条quote答案语义**，以复用其可复制原文与固定公共规范化。纯ID方案保留为未选CPU对照，不能拿JSON往返证明HP编号长尾可学。

## 原始资产及唯一选型

| 项目 | base（选定） | large（本次不启动） |
|---|---|---|
| HF固定revision | `aecb1376e5bd78db32ebc5c9deb257449b9e2b21` | `fb4ba84440d10e9b93fd626fb460683372329d4a` |
| 原生架构参数数 | **247,587,456** | **783,175,168** |
| 层/宽/FFN | encoder12+decoder12，768/2048 | encoder24+decoder24，1024/2816 |
| FP32参数+梯度+两Adam状态 | 3.689 GiB | 11.670 GiB |
| 8k单encoder注意力矩阵FP32 | 336 MiB | 448 MiB |
| 16k同矩阵FP32 | 1056 MiB | 1408 MiB |
| 当前资产判断 | 原PT文件990,446,387B；规模与完整untied架构一致，实际加载仍必须零missing/mismatch | 配置要求untied LM head；发布safetensors实际只有750,276,096参数，**没有`lm_head.weight`**，精确少32,899,072参数 |

计数按固定config与HF4.50.0原生源码中共享输入embedding、独立LM head、两stack/relative-bias逐项求和，尚未实例化模型。矩阵只是单个中间张量，**不是整模型显存峰值**；TGlobal仍含随长度增长的全局块注意力，不能称完全线性。详细公式/尺寸见[capacity-comparison](evidence/capacity-comparison.json)。真实8k/16k耗时与峰值目前未测，不能用这些静态数值代替。

large具有更多容量，原论文部分任务也优于base，但表中任务的长度/微调条件并非全部相同，不能据此宣称当前87患者必优。这里选base的首要依据是large当前PT配置/权重头存在静态不一致，根因仍待核：只读取73,760B safetensors JSON头，零tensor payload，已确认缺失；不能默许随机新头或改`tie_word_embeddings`掩盖。原large Flax文件3.132GB可能保留完整头，但转换和恢复尚未核实，另行处理才能重开，不下载两套试分数。base实际加载也必须先核零missing，不能仅由文件大小宣称权重完整。

两款原始模型均Apache-2.0。Google原论文§4.1明确仅用C4预训练、4096输入/910输出、PEGASUS式无监督主要句子生成；所选是**原始预训练checkpoint**，不取PubMed摘要微调或外部phenotype人工标注模型。原论文与项目报告16k微调/QA可用；config中的`n_positions=4096`是预训练元数据，原生LongT5用相对位置与局部/全局块，没有4096绝对位置embedding硬截断。不能把“支持16k”写成“原始权重已16k预训”。

ModelScope同名仓库实际可访问，虽然标记`USER_UPLOAD`，base `43e71100fb6bb97afaaff5de08a13aa6210dfb6c`的config、tokenizer.json、spiece分别与原厂固定HF版本逐SHA一致；PT文件发布LFS SHA也同为`288fc9c6abd72d86f6e05ed3f3691628f11bf8ab45d69424eb31fc94000e2e34`。后继优先该ModelScope固定revision并完整回读核SHA。large同名MS配置/SP也一致，但不解除缺头问题。所有下载/元数据来源在[sources](evidence/sources.json)，没有把发布SHA记成已下载权重SHA。

BioBART-large原配置`max_position_embeddings=1024`，不能直接覆盖本任务全文；PubMed摘要BART衍生品也不能仅凭名字推定长上下文。Flan-T5-XL约2.85B、dense encoder注意力，无LongT5同等长输入效率；相同10k/16k输入在A30上成本更差，本次不选。不存在“把max_length改大就解决BART位置限制”的计划。

## 忠实供给与原生token成本

固定继承原G `targets-v2` manifest SHA`85979cdc…` 的87 train与39纯输入，原train32 SHA`38c353ea…`，F1输入SHA`962b76a3…`，无文献交叉。将原两条system/user内容用两个换行连接交给encoder，所有全文passage、患者表与定位原字节保留。目标**不重新挑anchor、不做新teacher、不从818 ID替换/截取**，逐条使用原G答案801个weak quote triples，代表814个canonical keys、5空患者。原G已有4个incomplete患者/4个invalid relation label，目标不能声称覆盖原818全部v2原子；这4行保留，不私自补造或删除。

| 原生SP，含EOS | 未选direct ID | **选定原G quote** |
|---|---:|---:|
| train87 encoder总 / 最大 | 463477 / 9712 | **484678 / 9907** |
| held39纯输入encoder总 / 最大 | 213904 / 10754 | **222706 / 10961** |
| train decoder总 / 最大 | 5997 / 298 | **12943 / 658** |
| 语义供给 | 原v2 818原子，数字ID长尾风险 | 原G 801可复制quote/814canonical，保原监督局限 |

两套均完整87/39、未截文献；最终只选quote。旧direct ID供给`prepared-v1/`和计划`probe-plan.direct-id-not-selected.json`保持，不进入模型。新`quote-prepared-v2/`保存每行原答案、可逆传输答案、label IDs各自SHA，见[evidence/quote-supply.json](evidence/quote-supply.json)。

原G第18行有两个`<`：直接nativeSP会UNK；标准JSON `\u003c`也因反斜杠UNK失败。按root明确的可逆序列化准则，**仅把目标中这两个原生UNK位置映射到词表已有`<extra_id_0>` ID32099**，其余原SP IDs不变；decode pieces先恢复字面`<`，再交原G严格JSON/quote parser。没有新token/embedding/quote，也不改变原JSON语义。CPU实际87/87原答案字节及JSON逐项顺序完整恢复，双SHA及labels SHA均保存。传输token序列有变化，这一额外工程变量明确披露；不是偷偷修文字。

固定原厂`spiece.model` SHA`d60acb12…`与SentencePiece0.2.1；EOS1、decoder-start/pad0、UNK2。SP自身`pad_id()`为-1，模型pad取config0。发布fast JSON在33/126 direct-ID输入发生Unigram切分差异，已定位但未穷究根因，后继不用fast重建。**输入原文完全不改**，原G输入原生UNK包括`<`182、`~`45及JSON患者表`{`608/`}`608次，必须披露，不能称无损token化。quote目标经上述专用codec可逆，不把输入UNK静默替换。

## 下一步仅一次工程，不是正式训练

[probe.py](probe.py)已准备、未执行：原厂base全参数、FP32权重+BF16 autocast、原生encoder/decoder非重入checkpoint、AdamW。先真实free UUID/无进程门和300秒启动有效期，再四个完整训练案例：5571×658最大目标、7733×213右补到8192、9907×3最长输入、同完整输入右补到16384。padding mask=0，不截文本、不增加临床内容。前两例各真实Adam一次，后两例在已有Adam状态下做FB；保存loss/梯度有限性、实际8k/16k FB秒数、峰allocated/reserved，再16-token训练例工程解码。所有probe状态弃掉，无checkpoint恢复声明。

.100上单base资产/探针已由root授权，正式训练未授权。共享预算600秒+TERM30，排他输出目录、不自动重试；`.100 A30 GPU0`只是候选，root排程时SCST必要Sap优先，须真实空闲门。没有申请/启动正式87训练；需先取得每例实测与目标环境版本，再决定正式配方/预算。没有实测前不能填写训练分钟数。入口、原件、精确将来命令见[README](README.md)；此probe不运行held推断或质量。

主要原始来源：[Google LongT5](https://github.com/google-research/longt5)、[NAACL论文](https://aclanthology.org/2022.findings-naacl.55/)、[base原始模型](https://huggingface.co/google/long-t5-tglobal-base/tree/aecb1376e5bd78db32ebc5c9deb257449b9e2b21)、[large原始模型](https://huggingface.co/google/long-t5-tglobal-large/tree/fb4ba84440d10e9b93fd626fb460683372329d4a)、[ModelScope base](https://modelscope.cn/models/google/long-t5-tglobal-base)。已核机制地图与当前route索引；这是架构条件生成控制，不称新的集合监督发明。
