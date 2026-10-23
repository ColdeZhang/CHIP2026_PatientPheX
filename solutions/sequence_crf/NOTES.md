# CRF 序列标注研究笔记

只保存可复用的方法来源、工程经验和待验证假设；试验结果以[实验过程](experiments/E-20261023-009.md)为准。已确认跨方案坑回报主线程更新根 HANDOFF.MD。

首日结果见全局实验索引；第二轮不能只优化关系选择，首日实体固定时关系oracle总分上限 .7803；增强实体后上限 .805345，距 .8只剩很窄余量，仍需要同时改善覆盖和患者归属。这是gold-aware规划诊断，不是真实候选成绩。

- 内部折的患者构成/长篇多患者文献权重与原dev不同；两次患者规则/上下文增强均有内折收益但dev下降。必须同时保存分歧，不能拿一次提升代替完整证据。
- 预训练底座是训练缓存的一部分，除了训练数据、源码和超参数，也要将权重/config/tokenizer的SHA写入指纹；仅记录模型路径不够。
- 显式cohort归属只在数量与候选患者数相容时触发，并限制临床章节或明确our指代；泛称patients不能直接向所有列出的患者广播。

- 哈希映射不能以 basename 作为唯一键：不同子模块都叫run.py时会静默覆盖。角色名或唯一相对路径为key，并用同名不同内容的测试验证两个依赖都保留；已回报主线程。
- 神经HPO候选重排训练时，至少扣除本篇独有训练别名；否则训练样本正例通过精确别名进入候选，会高估泛化并学不到真正的语义判别。

- 同训练语料/相似NER结构的错误高度相关。新增跨度三路一致也只有30/112完全正确，不能因多数票就默认有足够精度；应先查相对现有实体池的边际正确率。
- Qwen候选归属提示虽明确排除泛化/阴性，zero-shot仍在固定fold0产生大量FP。先验证、记录，不以语言模型“更强”的直觉替代实测；小thinking对照应保持输入与预算可比。

- O 编号来源是官方“文章编号”schema，P 编号是标注顺序。完整编号匹配不能退化成裸数字搜索；当前家系分隔符等价化版本只是探索假设，四折不稳定，不升级默认。公开表格沿用更保守的唯一全串身份匹配。
- Qwen thinking 模式在固定输出预算下可能未输出最终答案。必须检查闭合think与生成上限；拒绝把推理中提到的示例JSON当最终决策，单独报告有效率。

- 公开原文表格可以匹配患者、含临床事实，仍不保证属于比赛的关系标注范围。首个严格表格增量31条全部为本地金标FP，不能用“事实正确”替代评分验证，也不能按金标漏/错例反向筛选表格。
- 二阶段模型的训练候选必须对该训练文献也是OOF：outer train48需再划3份train32→heldout16；不可拿train48拟合过自身的预测给SFT当候选特征。gold仅进入独立answer，不补候选、不改提示证据。
- SFT必须用真实聊天模板逐样本验证prefix token完全一致与answer-only loss，禁止静默截断。最长样本单步smoke和正式训练目录隔离，不把smoke权重续训成正式实验。

- Gold患者-HPO关系通常没有对应证据span指针。用该文任一同HPO正实体计算患者距离，只能给出乐观锚点可见性，不能当真实归属证据或共指金标。原文±500全局距离也不同于按passage裁剪的真实prompt窗口，应分别统计。
- 扩充患者编号不等于完整共指；全局patient anchors与候选局部marker也不同。未来共指应只读原文、输出精确可回定位字符串/患者ID，并固定同候选比较，避免把newspan、实体与关系同时换掉失去解释。

- 患者共指即使JSON可解析、text精确存在，也可能错在定位或身份：短he裸子串次数会落在the等词内；模型还可能把表型词当人称。应以完整词边界和唯一局部原文context定位，并单独审查患者身份，不能只靠合法坐标验收。
- 新共指短语不能隐式升级成全文alias：保留原始给定alias处理，generated mentions只使用核验过的真实坐标；旧alias广播另作可归因控制，不修改共享规则。

- LLM响应缓存复用必须比对完整输入row和实际模型/adapter/解码器/运行时身份；同row_id或prompt条数相同不足以证明可复用。变更行重推、未变行复用时逐行记录原响应文件SHA。

## GLiNER 结构方向：准确版本比家族名称重要

三nested训练诊断将下一步放在跨度覆盖：1364实体FN仅45同跨度缺ID；371/501关系FN来自候选外。411“有重叠”只是操作性分类，不能全部算可修边界；四context上限仅损失3个marker，不足以优先再做同池MIL。诊断已独立复算，完整范围/边界见training-loss-diagnosis与review。

优先研究 [GLiNER medium-v2](https://huggingface.co/urchade/gliner_medium-v2/tree/f7da09ddffab92deb4f2a815662d88b7e0d909d5)：官方卡明确NuNER数据，模型/代码Apache2.0，数据MIT；[NuNER论文](https://arxiv.org/abs/2402.15343)明确C4原文→GPT3.5自动注释，通用语义类别与span匹配不同于现有任务BIO。v2.1的pile-mistral卡仅license，不能因同GLiNER家族就继承v2谱系；v1含非商用许可，不作为首选。独立来源审查确认公开谱系，但全checkpoint训练/选择清单未公开，保留限制。没有使用含本赛train80的成品模型。

[冻结最小提案](experiments/P-20261023-GLINER.md)只先在train48按输入患者数/字符中位选择三篇，固定phenotype/.50、完整词与实际token预算、原passage坐标；只资源/定位pilot。后续若获分配才做同折Sap过滤与不重叠增量，固定阈值/规则比较，避免把多个改动同时包装为增益。可选2epoch自训已按后续冻结方案完成，结果见下文。

训练FN独立分层补充：908无重叠FN主要是短而未在相应train32见过的词面（中位2词，836≤5词，仅39词面已见）；仅6项>GLiNER的12词宽，411overlap中仅4项>12。大章节漏率无明显全章节硬截，CRF chunk跨界仅1项。因此GLiNER假设是通用类别条件span预训练的互补覆盖，不是更长跨度上限。该模型195.176M、3篇101窗3.058秒/0.830GiB已证明成本可控；未评分，不宣称候选优质。详见冻结提案和 `validation/nested-loss-review/miss-strata.json`。

GLiNER训练接口坑（训练前已拦住）：gliner0.2.22的SpanModel.loss位置调用与BaseModel._loss新增prob_margin参数错位，negatives=1被绑定为label_smoothing=1，原0/1目标全变.5；正logit2的正例梯度因此反向。实际A800库base.py SHA b3ef985596410c07a447f18f5e199339b97ca4f603441a9ff36f9242d949e5bd、独立数值probe SHA de238b865de637eea05dccaafb6458d0be0df1bdb9bed5cb15a3934a7fc9e1d1。不要把库版本/成功退出当loss语义正确；自有trainer显式从logits算BCEWithLogits+validmask，正负/全负/尾batch/无效span梯度检查通过，不改第三方环境。此前所有GLiNER推断未走loss，结果仍有效。


GLiNER两epoch监督控制表明通用span预训练可学本题边界：nested16纯span F1 .2773→.6600，预定融合后相对CRF .6910→.7006。但边际98TP/147FP仅40%精度，不能把单模型提升当最终得分提升，后续先冻结链接过滤及更长训练对照，不能在同inner16扫阈值。epoch2训练probe recall仅.705，有继续固定预算的依据；该inner16已用于路线开发，未来不得称新盲测。原2ep scheduler已走完，真实optimizer/RNG存在也不使追加8ep等同从头总10ep；若改总预算应明确fresh或重启schedule政策。
