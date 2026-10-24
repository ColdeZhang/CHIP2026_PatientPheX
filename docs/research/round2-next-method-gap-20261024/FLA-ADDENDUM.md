# 补充：FLA应作为独立backend评估，旧失败不改写

2026-10-24 UTC，有限只读补核。读H43、原GPU_RESULT、实际metrics/timings及GPU probe源码；没有安装、远端操作、GPU、训练、生成或评分，也不改变当前native作业。

**建议保留一次新的、独立且有界的前瞻诊断。** 目的不是把原失败改判为通过，而是判断当前私有FLA能否作为可接受的独立数值backend，在真实长输入上获得值得后续验证的速度收益。它是工程候选，不是另一个已经证明高分的方法；收益和实际任务质量均未知。

## 原证据支持的准确结论

原128-token完整9B logits relative-L2 .090575123、maxabs13.03125未过预设.01/.25；mean KL(native||fused) .000412400通过.001。单GDN forward/decode及9个参数梯度和9次cache结构／数值检查通过。原全模型gate采取这些指标联合通过，因此正确结论仍是**原合同失败**。

原logits未保存，teacher-forced、greedy结果虽在函数内计算过，却在保存／比较前被首logit断言阻断。不能现在补算中心化差异、声称实际是整体平移、声称生成相同，或把短synthetic数值当任务质量。没有完整LoRA梯度／更新终点。

单块warm forward native .003093秒、FLA .006341秒，实际更慢；fused冷forward/backward140.399秒含JIT。整模仅128-token各一次prefill/decode：native .131192/.851928秒，fused .077762/.935022秒，三次交替热测未开始。不能用prefill单点宣称端到端加速，更不能推断2704／5556／14390 token成本。

## 为什么raw logit硬门不足以永久排除新backend

数学上每位置加任意共同常数c，`softmax(z+c)=softmax(z)`且CE不变，raw L2/maxabs仍可很大。按位置中心化`z-mean(z)`或比较`z-logsumexp(z)`可以分离这种不影响概率的自由度。**这是指标性质，不是对本次差异根因的证明。** 低mean KL也可能掩盖少量关键位置的大差异；argmax在近似平局处可以因小扰动改变，后续自回归路径再发散。

所以新诊断应同时保存raw差异、每位置中心化／log-prob差异、概率距离、top1/top2 margin及实际生成，而不是事后只挑一个过门指标。旧raw门保留为原合同判定；若采用新独立backend的门槛，必须运行前明确用途、数值／性能停止条件及版本，不能在看到新结果后扩大容差。

## 最小新诊断建议（未执行、预算未批准）

1. 复用已经安装的私有FLA固定0.6.0、准确TF5.20.2／Torch2.7.1环境和同一Qwen缓存。新namespace、新绝对期限和实际绑定证据；只改变两个GDN函数，卷积保持native，不隐式安装causal-conv，不升级共享包、不修改活跃进程。A100的Torch2.6不满足该版本已核声明要求，不能把.100/A800已缓存包默认搬用。
2. 重复原128-token合成输入，先持久保存两个backend的**完整实际logits、dtype、shape、输入IDs和版本**，再计算／判定。可以同shape增加一次native重复，以区分本机原生波动，不能把其它R1模型的漂移直接当FLA容差依据。原失败仍独立保留；新测量不是恢复原丢失张量。
3. 对全部128位置报告centered L2/max、按位置KL/概率L1的median/p95/max、top1一致率及不一致位置margin。保存完整cached teacher-forced 8步的全词表logits／cache结构和各自greedy IDs；先raw落盘再严格解析。若关心患者集合，应另用纯输入固定真实prompt生成完整JSON，不能凭8个前缀token一致推断整条任务行为。
4. 选实际已有纯输入，按事前长度规则覆盖中等约2704、训练最长5556和完整全文最长14390（必须实核所选row真实长度，不能裁拼成指定数字冒称真实输入）。同adapter／相同模型权重、tokens、shape、decoder、cap和空cache，各backend独立prefill；不把native cache传给FLA混成测试。冷JIT与第一次计算单列，至少3次交替热测，中位prefill、固定decode、完整生成总耗时分别报告；逐次CUDA同步、峰allocated/reserved、合法schema/EOS/cap状态保留。
5. 大产物有边界：128×248320双BF16 raw logits约127.1MB；14390全位置双BF16约14.29GB，若实际float32则翻倍。完整短logits必须保存；长输入默认保存**预先指定决策位置的全词表**logits（含末位prefill与cached续写），以及完整raw生成和全量索引，而非默认写全位置14GB。这不构成所有长位置的数值证明；若确需全位置，须事前留存储预算并单列。全部留仓库外，不入Git。
6. 本阶段只回答推理backend可用性与成本，不做fit／质量。如果长输入端到端热耗时没有实质改善（可沿原建议至少1.20×作为预先讨论门槛），或出现非finite、实际fallback、cache不完整、持续错误增长／格式失败，则停止扩大；三条文本看似合理不能替代计量。输出不完全相同不自动等于质量下降，但只能作为不同backend的输出，须未来另定完整F1质量控制，不能静默替换当前结果。

若长期目标是加快**训练**，推理通过仍不够：FLA chunk有backward，recurrent backward未实现，训练须use_cache=False。未来还要单独有界地测真实长样本的forward/backward、Adam已有状态、累积梯度、真实更新与端到端吞吐；不凭单块通过或prefill提速启动全文正式训练。不把这项额外检查塞进当前运行预算。

## 优先级

当前训练继续native。该新诊断可以放在现有完整终点之后的空闲兼容GPU上，避免再把长任务主要时间花在不知是否值得的实现假设上；应先取得长输入**实际**速度，再决定是否为fresh四轮或后续全文训练采用它。FLA诊断和四轮实验若同时改变backend与时程，将失去解释性，应分别结案或明确组合变更。现有证据既不保证速度，也不保证分数，只有一个值得补齐的测量缺口。
