# 901之后的关系机制盘点（只读，一页）

2026-10-24；目的：区分已关闭配置与真正未试机制，避免把旧负结果换名重跑。仅读取现有source/report/metrics并核SHA；无gold、MCP、NN、fit、重评或官方操作。以下预算是后续提案，未启动。N/R为同一旧Qwen adapter的正序/逆序；P为另训的patient-first adapter，不能称三独立基座。

| 已有实际证据 | 结论及边界 |
|---|---|
| N∪R：F1 .660942003→.662472447（micro下降），F0 .695443948→.704687440；原dev旧体系 .738493753→.746081965，新增15TP/8FP | 测试时排列有互补；R-alone在F1负。原dev A30→A800硬件也变，不能全归因顺序。见[逆序F1](../experiments/reverse-json-order.md)、[F0](../experiments/reverse-json-order-f0.md)、[原dev](../experiments/reverse-json-order-dev.md)。 |
| N∪R∪P：F1 .663841821（比N∪R +.001369374，micro下降），F0 .709556743（+.004869303，仍低rules .711353522） | 两折固定门通过，但只是小幅互补；原dev patient差集仅6条=3TP/3FP。901 .754361398来自新版support最佳追加这6条，macro下降；不应推断多数票或更大收益。见[三方并集](../experiments/relation-triple-union.md)、[901合成](../experiments/support-plus-patient-six-delta.md)。 |
| 四逻辑epoch O/O/O/O vs O/R/O/R | **只有CPU可行性**。实际训练是从同旧epoch2权重、新Adam/LR开始的新两轮O/O vs R/O：F1 .658421563/.645528610，均低旧 .660942003，关闭该配置。不能把它写成四轮原计划已运行，也不建议仅为补齐四轮标签重训。[实际终点](../../../../round2-case-retrieval/solutions/case_retrieval/order_warmstart_stage/RESULT.md)见下方绝对定位。 |
| 旧多数票/stacking | LR/rules/Qwen8B多数 .692310632，LR/rules/SFT8B多数 .709202926，均低同池rules .711353522；五信号LR .697656525也低强基线，已关闭。它们不是N/R/P多数。[旧多数记录](../experiments/E-20261023-009.md)、[五信号结案](relation-stacking-five-feature-control/README.md)。 |

**连续分数必须分清两条。** retrieval的`continuous-relation-confidence.md`所提独立yes/no verbalizer仍仅CPU预算：4127pair/4,144,716输入tokens（旧6.39倍），未计算logits；已有raw无logprobs，不能从耗时/长度造置信度。另一条JSON合法患者子集条件似然却已实跑：98分支pilot后扩到同12提示的530分支，158pair固定边际≥.5从greedy34TP/7FP/4FN变24TP/21FP/14FN，关闭该原始分数政策。不能统称“连续LLM置信从未试”，也不能将这个负结果等同独立yes/no无效。[置信判断](/workspace/patientphex-worktrees/round2-case-retrieval/solutions/case_retrieval/research/continuous-relation-confidence.md)、[实际条件分数结案](/workspace/patientphex-worktrees/round2-case-retrieval/solutions/case_retrieval/research/conditional-subset-likelihood.md)。

**未找到实际终点的具体机制**：第三个固定候选排列；N/R/P三者至少二票的关系集合。检索范围为三路线当前研究/实验报告、相关源与结果索引；这是库存结论，不是已证明不存在任何历史文件。多数票会删除原并集关系，适合检验precision能否抵消recall损失；三路错误相关，收益不作保证。

**最小证伪建议**：优先一个纯CPU控制，用已冻F1/F0的N、R、P完整缓存逐patient字面关系≥2票，实体/非关系字段不动；两折预测共同freeze后各一次v2，复用旧指标。30秒+TERM1、0NN/0fit；事先要求两折Score均严格超过各自N∪R∪P且macro均不降，同时并列F0 rules强对照；任一失败关闭该配置，不扫票权/阈值、不直接迁dev。第三排列只保留备选：固定循环左移1并同步CID，先CPU逆映射/原文/token合同；若另行选择，F1原80单次GPU600+30（原逆序实测282.4秒仅作预算依据），不同时试随机排列。独立yes/no先做冻结模板/native token契约和极小前向成本探针，真实吞吐未知，不把4.145M全量承诺进短预算。

精确证据定位与SHA-256：`S=/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf`，`R=/workspace/patientphex-worktrees/round2-case-retrieval/solutions/case_retrieval`。

- `S/experiments/relation-triple-union.md`：`4b150989dcd3735f1df96eb27d9c37f6a1c84376a5bb70c6c539020483d6f4ba`；该页链接实际F1/F0输入及输出ref。
- `R/order_warmstart_stage/RESULT.md`：`93a552780790fe87780f65de745692e43cb52d86cfb56328bda0f66269ae038f`；原四轮CPU计划`R/research/candidate-order-training-feasibility.md`：`ef8876c04f5b8bf031404a568ed6e0b8c28e5cc35332fc4a3c0ec83dddc23a75`。
- `R/evidence/round2/continuous-relation-confidence.json`：`984799ee2fd2cf892a4ef3d630fe470372d7dbea4278457bb3ad1b4847408737`；`R/evidence/round2/conditional-all80cid-result.json`：`8a41c2913bd5ae404805b5a56836ec67a7ce67885d4d5a204cb0494c17fd5c29`。
