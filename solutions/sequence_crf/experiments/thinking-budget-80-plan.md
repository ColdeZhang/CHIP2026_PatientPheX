# 固定80条思考预算：CPU准备完成，因成本暂缓

2026-10-24，`E-20261023-009-thinking-budget-forcing-80-fixed2304`。**主线程明确暂缓，不启动GPU，不另换更短cap凑成本。** 单条人工边界[格式PASS](thinking-budget-forcing-readiness.md)保留；完整80的效果未知。本任务只读原输出/纯输入，未读任何gold或targets，没有质量运行。

唯一政策冻结为 **思考≤2304、final≤512**；若以后选择执行，建议同原生A800、旧SFT32一个adapter，整job **14400秒+TERM30**。代码与实体候选池不变，仅替换关系推断。当前未分配该GPU资源，A800正在运行的warm任务不受影响。

## 原行为摘要的必要校正

原4096实际生成IDs经固定词表ByteLevel逐token逆码，与完整原response bytes exact。使用“最早包含事件完整末字节的1-based生成token”计数，不靠截断文本重新分词。

| 观测事件 | 生成token位置 | 能证明什么 |
|---|---:|---|
| 八个CID字面都曾出现 | 1479 | 包括开头范围描述中的C000/C007，不能当作已逐项决定 |
| 初轮七项明确列表决定齐 | 1721 | C003仍在讨论 |
| C003首次完整肯定映射句 | 2214 | 该句明确支持映射，不评其正确性 |
| C003首次明确列表决定 | 3529 | 此时才齐八项列表决定 |
| 首次连续完整八项复核结束 | 3704 | 只说明该文本过程，并非必要计算量 |

因此旧“前半已明确全部八项”的表述应收窄。char7268的Refining标题也不能单独证明循环从那里开始。旧raw和`behavior-diagnostic.json`保持冻结，新证据作为修正附录；[完整证据定位](../evidence/round2/thinking-budget-80-policy.json)中的coverage/report/source可追溯，实际分析源码快照另保留。

2304在该条2214位置后留90token余量，允许检验有限思考后强制出答案的假设。**一个样本的位置不能证明80条都已经思考充分，也不能保证格式或正确率。** cap依据这条已见F1输入的无标签行为诊断制定，不能称完全盲选；没有用F1质量分挑预算，也不扫描多个cap。

## 80条纯输入与成本

旧80/16篇/587概念/1557pairs不变，实际thinking=True输入总250536tokens，均3131.7，中位3140，最大5689。每条最多8CID/7PID，最密合法compact JSON+EOS只需242tokens，512可容纳所有合法集合的紧凑表达；模型仍可能浪费预算。最大人工重放prefix7996、含final最大8508，低于14000输入cap和262144上下文上限，无截断需求。

最坏新增token合同为`80×(2304+512)=225280`。以旧A800单条4096/186.896秒作线性代理为10279.28秒；再用本次A30的6.213秒×80以及32.337秒前置作粗略prefill/加载余量，总约**10808.67秒，3.00小时**。A30短54-token测量含prefill，不能当持续thinking吞吐；跨硬件成本也只是计划代理，非真实上界。4小时硬预算含加载、全部80双阶段；若届时超时，仅保留部分工程产物，不以部分数据评分。未来实际显存/时间尚未测量。

思考cap不再下调；先推进更便宜的patient-first CPU与最小更新成本核。这个决定不否定thinking方法，只说明现有证据未支持先投入约3小时。

## 两阶段状态机必须保留的区别

1. Stage A以`enable_thinking=True`开始，在首次248069、真正EOS248044/248046或2304 token cap结束；保存真实IDs和停止原因。
2. **自然248069**：保留真实close token，直接以原prompt+完整Stage A作为final前缀，不再人工加close或换行；新final可以自己生成空白。自然结束token恰好落在cap时，按token种类优先，不误判为强制分支。
3. **真正EOS但尚未close**：该行明确失败，不再加边界续写，不把思考内JSON当答案。
4. **只有预算耗尽**：仅此分支追加人工`[198,248069,271]`，原思考IDs不动。无`Wait`或`Final Answer:`，不扩大cap。
5. Stage B至多512，以真正EOS结束；presence历史为Stage A模型IDs与新suffix的并集，排除的是原prompt/人工边界来源位置，不全局屏蔽相同token ID。每行在Stage A前按原行号seed一次，两个阶段在同进程保持当前RNG；不冒称恢复旧失败run的RNG或KV。
6. 只解析Stage B新suffix，去掉末尾一个真实EOS，其余special tokens不隐式吞掉；CID键严格等于该行code_map（末块可能少于8），PID值合法且无重复，完整JSON。异常role/重复close/中途无合法停止原因须明确记录，不能靠重新套模板修复。

全局OOM/非有限/运行异常/超时首失败停止；单行格式失败记invalid/空集合、无旧模型fallback，不补跑。若未来完整80获准并完成，两份fullpred统一freeze后才一次F1 v2，匹配旧nonthinking80基线`.6609420026531936`；保留要求Score提高且macro不降。这里没有执行该质量端点、不迁原dev、不上传官方、不并集第二adapter。

冻结[机器政策与来源](../evidence/round2/thinking-budget-80-policy.json) SHA `231c42c2…`。输入普查复现源[census_80.py](../thinking_budget_forcing/census_80.py)，外部`research/thinking-budget-forcing-80-plan/`保存完整80行统计与token覆盖证据。该任务到此以“准备完成/暂缓”结案。
