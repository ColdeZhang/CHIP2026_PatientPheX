# 患者交换 DPO：准备结果

**已完成，固定配方负结案。** 唯一内部F1主端点从 **.701325377212172** 降到 **.7009053889090149**，Δ **−.000419988303157**。完整80fresh且80/80严格解析、0弃权；新v2=1、旧v2=0。没有进入F0、full64、原dev或官方提交。本线未产生原dev提升。实际证据见 [final-result](evidence/final-result.json)。


**值得做一次有界控制。** 新信息来自训练金标中“同一表型属于患者 A，而不属于患者 B”的成对排序，不增加模型、候选或文献。当前只完成数据、数学和真实微型模型 CPU 检查；没有运行大模型、训练新适配器或评分。首要风险是只有17篇可提供这种对照，合成负例也未必是模型真实犯错的难负例。

## 新机制及准确父模型

选择已完成的原生 `Qwen/Qwen3-8B@b968826d` train32 epoch2，而非正在跑的 full64 模型。基座8,190,735,360＋LoRA43,646,976＝**8,234,382,336<10B**。父适配器174,655,536B/SHA `543c82a5…` 已随旧F0迁移到 .3，私有TF4.51.3/PEFT0.15.2环境也已实际运行过；本轮尚未重查远端大文件，执行时需核实际父文件身份。无新大权重下载。

查重范围为主机制地图和各活跃方案的源码/记录。旧 query-slot 是共享prefill后两类CE；value×2、额外SFT、患者均衡loss只奖励金标续写；PCS竞争loss是跨度袋上的多标签分类。均没有以冻结SFT为reference，对完整chosen/rejected输出做相对概率DPO。检索命中MedGemma生成配置里的默认 `dola_layers`，不是已执行DPO/对比解码。

DPO采用 [Rafailov et al., 2305.18290](https://arxiv.org/abs/2305.18290) 原始sigmoid目标，参考[作者实现](https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L45)；Apache-2.0。作者代码明确完整回答**求和**log概率，β.1是其README起点之一。论文是通用偏好对齐，不提供本任务涨分保证；本项目的合成交换与原研究的人类偏好数据不同。

## 实际训练供给与分词

原124行、32篇F2/F3的输入和金标答案（`cb7f67e…`）保持。对每行枚举同CID的positive→同文真实label-negative患者交换，按固定SHA种子20261024选一个；只替换该CID的一位患者，关系条数不变，其他正患者及其他CID答案保持。这里的negative是**赛事训练标注中未关联**，不是声称临床事实不存在。

- **74行/17篇**有合法交换，共1149可选交换；每行只取一个，无多负例数/随机种子扫描。
- 50行不能交换，其中47为单患者；不造假负例补满124。选中的14行有多患者共同阳性，其余阳性全部保留。
- 实际CPU **1.917s**：原124 chosen的完整input IDs/labels与已核原生缓存逐项相同。新74双序列共519,954 tokens，各侧259,977；各侧4672监督tokens，最长5534。
- 70对回答token数相同；2对少1、2对多1，总量相同不等于逐对等长。所有prompt IDs一致，EOS151645＋newline198和全串往返相同；不能拆分tokenize后拼接。
- 魔搭准确revision下载的tokenizer SHA `aeb13307…`与父模型一致。本次原生Rust0.23.1完整新reject已核；目标HF4.51.3/tokenizers0.21.2对新增reject的实际接口复核仍是正式前的小门。

没有读F0/F1/原dev金标。祖先train32/F1隔离沿用父Qwen3完整审查及原124候选来源；未引入新的训练词典或held派生提示。全量训练记录/编码在仓外，Git只存脚本及摘要。

## 固定最小配方及内存实现

从父LoRA warm start，建立新Adam，不伪称恢复旧优化器。固定**一轮74对、累积8对、10次更新、尾2对、warmup1、LR1e−5、β.1、weight_decay .01、clip1、seed20261024**。禁用dropout，保原r16/alpha32/七线性目标、BF16骨干/监督logits FP32；不截输入，不加SFT项/长度归一/label smoothing。禁用原LoRA .05 dropout是明确的附带变化，用于确定性串行梯度，不能称只改一个loss符号。

冻结父模型一次计算148完整序列的reference log概率，随后卸载；新policy无需与第二个9B同时驻留。每对先在当前权重no-grad计算两个分数，再令 `c=β·sigmoid(−β[(pc−pr)−(rc−rr)])`，分别反传 `−c·pc` 和 `+c·pr`，每次只保留一个序列的激活；一组结束前不更新参数。记录真实DPO loss，不把用于反传的代理表达式值当loss。

真实 .3 CPU合同 **0.245s**、wrapper rc0：Torch2.7.0/TF4.51.3/PEFT.15.2，微型原生Qwen3＋r16实际调用通过。indexed/full shifted-sum差0；28/28 LoRA联合DPO与串行梯度最大差1.30e−8，全部非零，base无梯度，一次真实tiny Adam通过。另NumPy有限差分21坐标最大差8.37e−12。**这些不是BF16大模型或显存证明**；actual no-grad/grad前向须逐对先保存差异再按预写1e−5相对/绝对容差核验。

原Qwen3 SFT248行/806,302tokens实际686.3s、最长5534、峰21.18GB；80全文推断504.5s。按当前519,954双序列和额外coefficient前向，单轮训练粗代理约590s、reference约148s，均未测新配方且未计新host/加载/持久化。建议固定probe240、reference600、formal1200、infer1200秒，各TERM30；只能在实际工程门和成本门通过后启动，不能把估计当完成。序列串行避免两份激活同时存活，Adam后的峰值仍须真测。

## 唯一质量协议

主端点固定 **`B471 ∪ 至少两票(G, S, Q_DPO, C)`**，以新Q替换旧Q，允许修正旧Q造成的关系错误；全部B471、Task1、patient及其他对象保护。原四pure计数G282/S383/Q416/C395，实际重建得到原B4全部524关系。非关系对象和B471前缀相同，但排序/序列化不同，**不称重建字节相同**；若新结果关系集合不变，直接继承原B4文件，0次新评分。

全80fresh原始响应、严格解析、完整预测冻结后，仅1次公共v2 all/text，与B4 **.701325377212172**比较，并披露新增/删除及micro/macro。没有Q-alone、整并集、β/epoch/患者筛选；严格正才考虑固定F0。内部F1是开发集，原dev最好.769235669942仍未改变。

记录初始PT父模型建议被原生Qwen3替代，以及“只追加”改为固定Q替换的理由；均在GPU及heldgold读取前完成，不是失败运行或事后挑公式。执行入口、精确SHA与未执行门见 `candidate-plan.json`、`evidence/readiness.json`。

## 实际执行状态（14:25 UTC）

目标CPU148完整接口已4.580s通过。root把准备时预算建议替换为ref900/probe600/formal1800/infer1200各+30；原建议保留为历史。原reference148实际完整133.745s；首probe在12.449s的chosen backward因checkpoint元数据不一致停止，原chosen no-grad/grad logp均−.2504003942，未失败于预写数值门。正式训练/质量均未启动，109原成员及SHA完整保留。

CPU BF16＋同nonreentrant checkpoint单cache标志对照0.355s：开启缓存时梯度存在/finite门失败，关闭时两replay差0、28/28梯度和真实Adam通过；CPU没有复现CUDA同一CheckpointError，因果范围有限。恢复只关闭autocast权重cast缓存，保持原数学/数据/阈值。原148 reference按旧plan965b322d＋parent543c＋encoded9528绑定原字节fece4e82复用；新plan34cea8c6/attempt14:25:48沿**原probe TERM14:31:00.518905**，不重置时限。实际完整probe通过才以fresh543c/newAdam正式训练。

## 最终结果与决定

保持B471及所有Task1/patient对象后，524条融合关系的变化是新增2条（1TP、1FP）、删除2条（2TP、0FP）。micro .749042145594→.747126436782，macro .641320427466→.641556183066；Task1分项完全不变。轻微macro收益未抵消micro损失。新Q的关系数由416降到387，但没有额外Q-alone评分，不能宣称其独立精度提高或下降。

完整正式74对/148序列/519954tokens/9344监督、10updates实际555.637s完成；504适配器张量、504 Adam状态/1512张量和所有RNG完整持久化/回读一致。新adapter SHA75518d93、state d31283d6。80推断实际518.519s，服务器14:54:39已4MiB/no apps释放。全部原始记录、来源与预测在仓外，Git只存小摘要。

按事前严格正增益门，**关闭这一个固定74对/β.1/LR1e−5/一轮配方**，不追加阈值、β、epoch、seed、融合公式或第二质量端点。这不是对全部偏好学习机制的否定；本次证据不足以支持继续占用下一折与全量训练资源。缓存修复的工程结论仍有效，原失败与实际通过证据均保留。

最终统一入口：[结果](evidence/final-result.json)、[完整回读备份](evidence/backup.json)。95成员/540112B小包cf82e190已远端落盘和实际全量回读；独立终审74e4f195通过，后来回读receipt1a125633只追加备份事实，不修改原审查。
