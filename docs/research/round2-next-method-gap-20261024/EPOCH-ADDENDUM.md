# 补充：两轮不是已验证最优；更长训练已有部分反证

2026-10-24 UTC，按root追加要求进行有限历史核查。未读原dev gold、未训练／推断／评分，当前α=.5两轮不变。

**结论：两轮最初是有界首版配置、随后沿用以保持比较，没有查到“本任务两轮已足够／最优”的成熟依据或轮数选型证据。fresh四轮统一学习率时程尚未发现实际终点，可以作为未来一个低置信度、明确的新控制；但不能说延长训练完全没试过。** normal已经做过旧两轮权重＋重置AdamW/RNG/LR的新两轮O/O，实际累计四轮曝光且质量负。更长训练不能根据loss下降被预判为增益，也不应被一次控制的“本次不扫epoch”永久禁止。

## 1. 查到了什么

本次按文件名仅扫描五个指定实际run子树的`training_complete.json`，共11件，另读三活跃路线关键记录／真实trainer／旧训练统计。完整清单`epoch-inventory.json`。这是有限可定位库存，不是整台服务器所有历史不存在的证明；没有扫描或读取gold。

| 配方 | 已完成训练／质量 | 对四轮判断的意义 |
|---|---|---|
| normal JSON train32／48／64 | fresh各2轮，32／52／68 calls；train32实际1149.460秒 | 未见同recipe fresh统一四轮终点。初始9B提案复用8B闭集任务，并先按204×2测预算；未引用两轮最优证据。 |
| normal warm-start O/O | 旧SFT32 epoch2权重＋新AdamW/RNG/LR两轮，新增32 calls；F1 .660942003→.658421563，310TP73FP→299TP86FP | 真正有过更多训练的完整负结果，但包含重置优化器／时程，不能归因纯epoch或证明所有更长训练无效。 |
| normal warm-start R/O | 同起点／新增32 calls，F1 .645528610 | 顺序增强＋额外阶段的负结果，不直接测试fresh统一四轮。 |
| 原fresh O/O/O/O vs O/R/O/R | 仅CPU准备64 calls／4warmup表，随后被上述warm-start方案替代 | 不能把计划当已执行；也不能忽略其后继warm-start结果。 |
| patient-first train32／48／64 | fresh各2轮，32／52／68 calls；train32实际1141.720秒 | 未找到patient-first更长epoch完整质量。不能拿normal O/O完全替代这一目标的验证。 |

Biomed患者CE的4epoch、PhenoTagger的100epoch、pair-query的8epoch属于其他目标／模型，不能算9B闭集SFT的轮数控制。固定槽、value×2、public212均另有两轮终点，不是原normal／patient的更长训练。

## 2. 训练曲线能说明的边界

既有在线epoch mean：normal32 .152564→.065690；normal64 .114268→.060500；patient32 .2666083614→.1188598782。它们是每行token mean再group mean的在线训练统计，epoch内参数持续变化、样本组不同，不是同一冻结模型的训练集／留出评估。patient与normal输出序列和条件分解不同，也不能因patient绝对loss更高就认定它更欠拟合。

normal64固定12条已训练提示的真实自由生成自检F1 .883721、7/12集合完全相同，证明关系学习链路有效；其余错误不能区分优化不足、输入／监督歧义和生成曝光偏差。该自检只否定“关系几乎没学所以一定需要加轮数”，没有证明两轮是最佳泛化点。

原patient源码明确`get_linear_schedule_with_warmup(optimizer,2,32)`和`range(completed_epochs,2)`，旧normal同类。2轮终点scheduler与optimizer LR已到0；旧48实际state核验last_epoch52/LR0。完成点是已结束时程，不是尚待第3轮的暂停点。

## 3. 一个值得保留但不立即启动的前瞻控制

**优先级低于当前两轮终点；若未来有空闲资源，患者方向的fresh固定四轮比再做一次normal warm-start更有区分力。** 它未有更长实际终点、训练接口现成，成本有限；代价是小样本重复曝光和更大的累计更新，可能只改善训练拟合并伤害留出。这个建议置信度低，不是训练诊断证明的必要修复。

最小方案为原patient train32的124条／原目标、原candidate与decoder、seed/r16/alpha32/dropout.05/lr1e-4/AdamW/.01/clip1保持，fresh公开base和fresh optimizer，从开始就固定4逻辑epoch。每轮15组8＋尾4，合496行曝光、64 calls、1,602,456完整序列token和18,776监督token。若沿原warmup_ratio .05，则预注册4 warmup calls和64总calls的单一线性时程；**不是从旧epoch2接两轮**。

这同时改变总训练曝光与LR horizon，须按“训练预算／时程控制”解释，不能称只把旧轨迹末尾延长。新方案在第32次scheduler.step后的LR为5.333333e−5，而旧32-call终点为0；因此新epoch2权重也不会等于旧2ep。首optimizer call仍LR0，实际63次非零LR调用。固定只评epoch4，中间checkpoint只用于恢复，不看held分数择轮。

若α=.5当前两轮取得新正证据，root也可在其完整结案后把“同α固定四轮”作为另一个前瞻候选，按实际2ep α作为对照；这属于由新结果触发的开发选择，不能假装本报告已预注册该选择。不要同时上normal4／patient4／α4三个臂，也不在运行中的α2方案改轮数。

终点应沿原F1完整80输入，比较明确匹配的2ep模型，并另固定其在强组合中的位置：例如只用新P替换既定N∪R∪P中的P，保留其他字段；单P可作诊断，二者需共同freeze后各一次v2。只胜过单P弱基线但强组合无增益，不应据此全64。与强全文 .676546355的关系以及不同实体模板必须分别披露，不能跨模板总分硬比。通过后才另定F0复验；不迁原dev选epoch。

same A800原patient2ep约19.03分钟，机械双倍为38.06分钟训练代理，加原80约3.25分钟生成和新增保存／传输余量；**不是实际4ep成本或硬预算**。A100、selected-CE或新工程版本均需按其实际吞吐另估。完整epoch状态保存策略应事前明确，不为节约副本冒充恢复能力。

停止条件：数值／覆盖失败则工程结案；完整F1固定端点不提升则关闭这一四轮配方，不扫3/5/6轮或对原dev择点。未来若有新的学习证据可以再研究其他策略，不能把本次停止条件扩成永久实验禁令。

## 4. H5候选诊断到件补充

owner的`candidate-shift-diagnostic/result.json`已实际完成0.383495秒，只读pool／selected、无gold／NN／LGB重predict／v2：pool16595→16499（126删／30增），public1769→1550；主selected删54／增12。54个被删selected的候选全部仍在新池，53项public字段改变（41失public、12分数变），另1字段同但仍可能受上下文或decoder竞争影响；12新增也全部public字段变。

这给出了确切特征／选择变化，**仍不能证明概率校准失准或OOF重fit会净增分**，因为旧／新全候选LGB概率未保存，且无逐改项正确性诊断。原20步固定控制保持负关闭。原REPORT的OOF条件后继继续作为备选，当前不据此扩大GPU；不重复owner诊断。
