# Nonthinking 双种子共识：只读可行性准备

2026-10-24。**建议保留为一个尚未实际做过的、小规模完整控制；现在不执行。** 未找到“同一 SFT32 Qwen3.5-9B、同一非思考全文输入、两个新增随机种子＋已有贪心”的已登记实验。这个结论限于本次历史范围，不宣称不存在未登记的外部尝试。全文仅读既有源码、metadata、纯输入与模型预测；无新 gold、模型加载、下载权重、GPU、fit、预测、v2 或仓库源码改动。精确文件路径/版本/SHA见 `sources.json`，成本和候选空间见 `census.json`。

## 已做过什么，缺口在哪里

三活跃路线共检查1,814个 `*.py` / `*plan.json`，18个显式 `do_sample=True` 文件均属于下表两类采样路线；没有发现本次非思考双种子共识。检索范围、命中行和SHA完整保存于 `history-scan.json`。

| 历史控制 | 实际身份 | 与提案的区别 |
|---|---|---|
| 固定多数票 S22/S23 | N、候选逆序R用同adapter的greedy，加另训patient-first P的greedy；F1 .663841821→.663437055、F0 .709556743→.699390541 | 多输入/多适配器，未随机采样；也不是当前全文G或强fullsupport |
| Qwen3.5 thinking采样 S24/S25 | 同旧32 adapter，`enable_thinking=True`，单输入seed20261023、T1/p.95/k20/presence1.5、4096上限；触顶、无EOS、未闭think，无质量 | 思考模式的一条格式/成本诊断，不是80×2非思考共识 |
| 强制结束thinking S26 | 复用已生成4096-token思考前缀，人工闭合后seed20261024续采样512 | 改前缀/结束条件的一条诊断，不能冒称全新非思考输入或第二完整种子 |
| R1 S27 | DeepSeek-R1-Distill-Qwen-7B，T.6/p.95/k0、逐行seed20261024，一次完整80 | 基座、模式均不同，没有本次同权重两seed投票 |
| native→FLA全文 S3/S13 | 同4f4815b6 adapter、同80输入；两者均`do_sample=False`；native A800与FLA A30 | 是后端/硬件变体，不是两seed。raw中的`native_response`只是同一生成ID的另一解码核，不是额外样本 |

## 可复用资产与成熟方法

公开依据已实际HTTP200缓存：[Self-Consistency](https://arxiv.org/abs/2203.11171)采样多条推理并聚合答案；[SelfCheckGPT](https://arxiv.org/abs/2303.08896)用随机响应间的一致性辅助判断事实；[Universal Self-Consistency](https://arxiv.org/abs/2311.17311)用额外LLM选择自由答案。这里有固定CID/PID闭集，可以机械对齐关系，无需增加一个LLM judge。**本提案是非思考结构化输出的采样共识，不是复现CoT论文；三样本很少，共享错误仍可能一致，文献收益不能外推为本任务涨分。**

选择已审FLA全文分支作为唯一建议入口，而非短normal：同一纯输入80行/16篇/39患者、587候选概念、1,557候选患者对；`expanded80.jsonl` SHA `2a95bad1…`，全部实际token映射已有S10/S11。FLA贪心raw `e73425c3…`、canonical `a7d8bc7f…` 完整80/80自然EOS、0触顶，适合作G。短normal虽然也是80行，旧raw没有自然EOS字段，不能把“未记录”说成“0个EOS”，也不将它按本提案的严格合同直接复用为G。native全文raw `fbc6f5be…`也完整，但**不把native G和FLA两个sample混成同后端对照**。

基座来源S2固定HF revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`，ModelScope等价运行资产revision `aa2aaf6060dd45e8ca24da2d37ffb0172eb985d6`；许可证仍沿原manifest的HF文件，不把同名MS LICENSE当字节相同。旧SFT32适配器：config `96564441…`、weights `4f4815b6c6c245c43dc976406942f43cecc7250266a0ec91ee6e1807fe63bfff`；实际总参数8,997,081,600，单模型<10B。原FLA完整执行使用 `/home/zhangyuheng/projects/CHIP2026_PatientPheX/sequence-round2/thinking-budget-final-v1/old32-adapter` 和已审private FLA-v1；本次只确认历史locator/回执，未远端fresh stat，未来启动前须核实际文件SHA和空闲卡。

缓存模型卡S1（`c5f5a8c2…`，832–838及1133–1143行）对**general nonthinking**一致推荐：T=.7、top_p=.8、top_k=20、min_p=0、presence=1.5、repetition=1，模板`enable_thinking=False`。卡中“nonthinking reasoning”两处参数彼此不同，不能拼成一个不存在的官方配方。

## 一个固定的最小控制，供root决定

- **唯一新变量**：同4f4815适配器、同80全文输入、同FLA/SDPA/BF16执行两次非思考随机解码；无新训练。G直接复用已完整FLA贪心。建议 `do_sample=True,T=.7,top_p=.8,top_k=20,min_p=0,repetition=1,presence=0`，num_return_sequences=1，max_new_tokens=1600。这里**明确不是完整照抄官方preset**：保留原结构化JSON任务的零presence惩罚，避免新增对重复CID/PID/JSON token的惩罚变量；不同时试1.5或扫参数。
- 固定种子流20261024、20261025；每行由`SHA256(seed|row_id)`导出非负63-bit seed，写入实际row记录，reset CPU/CUDA RNG。可依原输入长度顺序单行执行，种子不依赖完成顺序。两流各80条fresh，旧greedy不算新增随机样本，不因某seed得分差而替换。
- 原prompt、code_map、PID、模板和全部input token IDs逐行核既有SHA；不删例、不截全文，保留14,390最大输入和1,600输出上限。沿S5严格JSON解析：精确全部CID键、无重复键、值为唯一合法字符串PID列表；自然EOS才有效。原始token IDs和decode先持久化，再解析；触顶/错格式整行弃答，不补缺键、不修答案、不从think文本挖答案、不回退greedy。硬超时/异常导致160条未全覆盖则仅保partial，不出完整质量结论。
- 按`(pmc_id, 原concept字符串, patient_id)`对齐；保留compound/NIL原串，不拆ID增加票。给定行无有效输出是缺票，不是假负例。诊断可固定 `M=(G∩S1)∪(G∩S2)∪(S1∩S2)`；同一成员的重复项只算一票。
- **主端点固定 `B ∪ (S1 ∩ S2)`**：只有两个新sample都有效且一致的关系可以补入强fullsupport B，B全部保持，不做盲并集或删除。B为S28 `2a65056f…`，1189实体/471关系，已有Score **.6863389812716092**。实核其中**88关系不在旧SFT587概念输入空间内**，必须全部原样保留，不能拿prompt候选集过滤整个B。B的support是train48而采样器train32，均排F1，祖先沿S30已有审查；不是全资源同为32的因果消融。
- 在看新质量前一起冻结M与主端点、完整160 raw和所有弃答。只为这两个预写端点各最多一次v2，旧指标复用；如果主端点与B完整字节相同，直接复用B记录零增益。**不单独评两个seed、不给最好seed留后门**。主门仅Score严格高于B，micro/macro全部披露；否则关闭本配置，不扫温度/票数/seed。通过也只代表内部F1研发信号，不直接原dev或官方；是否F0验证由root另决。

## 真实成本及边界

| 现有80行同权重控制 | 实际输入tokens | 实际输出tokens | generation秒 |
|---|---:|---:|---:|
| 短normal/native | 250,696 | 4,644 | 258.474 |
| 全文native/A800 | 670,122 | 4,727 | 353.630 |
| 全文FLA/A30（建议匹配） | 670,122 | 4,720 | 521.774 |

两新全文pass确定需**1,340,244输入tokens**；以既有greedy长度仅作成本代理为9,440输出tokens，2×实测为1,043.548秒。建议预估**1,100–1,400秒**、一次绝对**1,800+TERM30秒**窗口（含加载/采样/保存），模型只加载一次，GPU由root排；这不是已实测采样速度。原A30 greedy80 peak reserved约23.24GiB、allocated19.98GiB，已成功支持最长输入；采样变长仍有风险。极端160×1600达256,000输出tokens，不可能用该经验预算保证跑完；预算不足保不完整状态，不能临时降cap、跳长行或延时。当前不申请额外权重下载、不改predict、不执行任何探针/模型/评分。
