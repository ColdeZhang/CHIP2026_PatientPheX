# 完整患者集合的自临界序列训练

**建议只激活四名训练患者、八次生成的供给与梯度探针，暂不训练完整87行。** 新机制是用完整患者的规范化概念集合F1，直接给序列策略梯度；当前只有CPU供给证据，尚不知道实际随机输出是否产生有效奖励差异。它优先于另换Task1长上下文编码器，因为当前目标更直接对应患者关系分项。

## 与已执行机制的实质区别

旧9B query-slot使用逐候选两列CE。其expected-F1后继仅在固定概率上选top-k，没有更新权重，见 `round2-case-retrieval/solutions/case_retrieval/operations/decision_target_design/expected_f1_control.py`。旧DPO从74行各造一个同CID患者交换，保持预测数目；偏好loss没有完整集合F1数值，见 `explore-patient-swap-dpo/solutions/patient_swap_dpo/{prepare.py,dpo_ops.py}`。患者均衡CE改样本权重，G的直接JSON/quote CE仍按token监督。已沿这些实际源和MAP查重，未见同等完整患者reward策略梯度；这是有界机制核查，不是全历史不存在证明。

Qwen3旧关系输入把32篇拆成124个CID块，单块F1不等于完整患者集合F1。G原配方刚好87条完整患者输入/输出，使用已训练d52e2e6c适配器和完整全文，更适合这个目标。奖励按当前部署的原G decoder、train32 alias、public HPO、b59保守FP16 SapBERT标准化后的概念集合计算，**不按quote字面重合率，也不凭训练gold跨度替代真实标准化**。

## 成熟来源和所选数学

Rennie等人的 [Self-Critical Sequence Training](https://arxiv.org/abs/1612.00563) 用测试时贪心输出的reward作REINFORCE baseline，无需学习critic。读取了公开论文页和成熟复现 [ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch/tree/4c48a3304932d58c5349434e7b0085f48dcb4be4)，固定commit `4c48a330…`，MIT。`captioning/utils/rewards.py` 对每个sample减同输入greedy分数；`captioning/modules/losses.py:RewardCriterion` 将优势乘输出logprob并mask EOS后位置。该仓库是成熟复现，**不冒称论文作者代码**；没有使用COCO数据、奖励库或其模型权重。图像描述的效果不证明本赛事会涨分。

也读取了[Shen等MRT](https://arxiv.org/abs/1512.02433)和[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)摘要，未另立两条配方。SCST只需每患者一贪心一随机，比多样本风险候选或GRPO组节省供给；不需要第二9B reference常驻。继承旧DPO的串行无梯度系数/梯度重放思路及H65缓存修复，**不复用其DPO目标或Qwen3前向接口冒充Qwen3.5已验证**。

固定提案：`A_i = F1(gold_i, sample_i) - F1(gold_i, greedy_i)`，baseline与advantage停止梯度。采样用T=1、top_p=1、top_k=0、repetition=1的完整categorical，重放必须计算同分布。每患者loss为：

`CEmean(original target_i) - A_i / (11884/87) * SUM logp(actual sampled suffix_i)`。

CE权重1、策略项权重1；11884/87是既有完整训练集平均监督长度，只作为**统一常数**调整量纲，不按每个sample长度相除。原论文期望reward梯度是序列logprob之和，所读复现有token平均版本；这里明确选择固定常数而非偷偷混用两者。小型CPU三结果穷举显示expected-SCST梯度与期望reward导数相同，有限差分最大误差4.38e-11；这不是实际9B反向的证据。

唯一probe新AdamW lr1e-6、β(.9,.999)、eps1e-8、wd0、clip1；四患者先在同一未变父权重下取得八条原始生成和完整reward，再八次串行完整反向（四sample、四原CE）与一次更新。所有dropout禁用，no-grad与grad的autocast缓存分开或cache=False，保实际EOS、不监督prefix、不补cap的EOS。probe状态弃掉，不接着训练。

未来如果完整训练获授权，必须每个累积组在当前参数下生成、算reward、重放，随后才一次更新。先预生成174条再训练整轮属于off-policy，**不是这个on-policy方案**。本提案没有授权完整87训练或held质量。

## train-only CPU供给的实证

[CPU结果](evidence/cpu-supply.json)使用原train32/87患者，保留v2的全部818个原子gold，包括旧G目标无法构造的4个literal，未删FN。5名患者gold为空。只复用已有624条train quote的原FP16 Sap数值与固定b59政策，没有新encoder/模型调用。

- 原监督target经过真实固定标准化后，患者F1均值`.994602503015`，76/87完全一致。**这是训练目标的闭环诊断，不是模型生成分数。**
- 82行删除一个可区分概念改变reward；71行加入另患者、目标未标注的概念改变reward。后者是训练标签意义的FP，不宣称临床阴性。
- 32行存在不同原文quote映射同一概念集合。集合reward保持不变，证明不能用quote字符串F1替代。
- 无新的随机模型输出，因此真实greedy/sample reward差异仍未知。人为扰动仅检验reward供给，不是探索性训练样本。

CPU首次调用Python3.12误加载系统3.13 NumPy，在读取训练数据前失败；换实际系统`/usr/bin/python3`后0.504秒完成。未重模型。初选四行把最长prefix空患者和另一个空患者同时选入；任何GPU前按既有shape轴改为 `[76,44,83,19]`，覆盖最长prefix、最长完整序列、最长监督和剩余最大prefix×监督。旧cpu-v2及选择保留，cpu-v3计算口径不变，详见[准备历史](evidence/preparation-history.json)。

## 一次最小探针与成本门

[冻结提案](pilot-plan.json)固定四患者：`5793168:P1`（空、8823输入），`12628480:OP1`（28概念、8811输入），`6134758:P1`（42概念、610原监督tokens），`12059664:O4`（31概念、6072输入）。每人greedy一次、随机一次，maxnew1600；最长实际输入加上限10423，无截断，无seed/温度扫描。原始IDs、EOS/cap和全文响应先落盘。

唯一供给门：四组中至少两组的**两条均合法完整**且规范化概念集合F1不同，至少一组gold非空；才继续新loss链工程检查。此门仅说明存在可学习的数值信号，不是临床正确率或预期涨分。无差异则不换患者、不重采样。不同于部署decoder允许部分坏item弃用，训练reward对非EOS/cap/不合法item一律给0，是明确的格式有效性门；有效空数组仍遵守空gold奖励1。所有原raw保留。

原G39 native A100实际10513输出token/1030.970秒，总1103.215秒；Copy39 A30无FLA的10361token/723.208秒是另一条件。刚完成的补全集合39仅1159token、33空回答，虽总123.479秒也**不能当普通G加速**。4×2×1600的12800输出上限，按原普通G生成速率代理1255.247秒；新反向、正常化、加载、完整保存留540秒，总代理1795.247秒。提议一次1800+30共享时钟，是有界尝试而非上界保证；超时保存原件并停，不增加预算或减样本。完整87轮的成本不由空回答的123秒推断，先看本探针实耗。

主体排在A800 GPU2当前F0-47实际释放之后，fresh UUID/空闲/盘门和H69 admission必需；CPU准备不能抢占已授权任务。已有624train cache之外所有必要新quote，都用原public Sap FP16/41099bank完整查询，安排独立已分配的.3或.100空卡≤180+30，计入主体共享时钟；不静默丢词/换FP32/跳过fallback。没有新大模型下载。

## 资源、隔离与限制

沿用已核ModelScope/HF c2022362的Qwen3.5-9B缓存、Apache-2.0；实际文本8,953,803,264参数，加43,278,336 LoRA为8,997,081,600。无需视觉组件，完整公开bundle原参数9,653,104,368加adapter也未过10B；各模型分别披露。原公开SapBERT109,482,240参数/090663c3冻结，只作既有normalizer，无新外部人工表型标注数据。允许资源范围继承官方README_EN§Participation Rules第3项，公有代码/模型/知识库引用齐备；本路线只增加数学和训练标签reward。

风险：uniform patient reward直接对应macro-F1，不能保证同时提高global micro-F1或总Score；CE anchor也不是性能保证。长期序列所有token共享一个advantage，方差大；训练中的歧义/不一致标签不会因为reward优化而自动正确；正常化弃答可能限制reward上界。若供给/工程通过，root再定完整训练与唯一强对照质量端点，不能用训练target接近1冒充达到用户0.8目标。
