# 第二轮研究索引：直接指导可检验的改进

本页由主线程维护，只保留共同方法依据；具体实现、实验过程和取舍由三方案各自记录。下载的论文/原文在仓库外，[来源与 SHA](round2-research-sources.json)可用于核验，不重复查站点。

## 主办方原始证据

已下载并读取[官方评测综述](https://arxiv.org/pdf/2610.29606) 3.3、4.2–4.3 节；HTML 路由未成功，PDF可用。论文前三名 B Score .7389/.7247/.7169 是官方不同测试集结果，不当作本地目标或可复现保证。

| 来源 | 可实施的思想 | 本轮落点 |
|---|---|---|
| 官方基线 §3.3 | Qwen3-8B 先扩充指定患者称谓/共指，再判断每批8个表型的归属，局部上下文前后各500字符；greedy，输出上限1800/2600 | 原有既定mentions上的关系对照已完成；全文LLM共指此前尚未实际运行，现由序列路线单独控制验证 |
| Agriv §4.2 | 多个PhenoTagger+监督跨度+HPO候选，SapBERT检索，两阶段LightGBM与上下文交叉编码器；LLM证据及位置特征经OOF融合 | 本体与CRF补候选/标准化，检索研究可学习关系判断 |
| 第二名 §4.2 | 多源候选投票、LLM处理重叠/否定；生成患者HPO集合与候选对分类的互补SFT | 检索零/少样本之后再决定是否折内LoRA，避免只扫阈值 |
| 第三名 §4.2 | 监督跨度、SapBERT、本体图特征；关系模型及LLM assertion复核，保留关系分数用于恢复 | 实体与关系使用可追溯候选池，避免最终实体阈值硬截关系 |

另已直接查看PDF第6页Fig.2，避免纯文本抽取漏掉图片中的提示词。共指阶段要求严格原文跨度、全局位置、仅指定患者、无歧义才收录，每患者每passage最多25处；不能把亲属、对照、泛指群体默认归给目标。归属阶段还排除否定、假设、待排除与仅家族史，不能凭疾病常识推断。实际实现若采用occurrence编号/P001等更稳格式，须明确是本项目改写而非官方原样复现。渲染位置与SHA见来源清单。

冠军公开[权重卡](https://huggingface.co/rigidk/CHIP2026-PatientPheX-Agriv) revision `7a98ca95683eb920b09cc9141cdf512dc5f1fa41` 声明所有自训标签来自官方80篇。可研究结构（BioLinkBERT-large、Qwen3-8B LoRA r16、单/多患者LGBM），但这些已训练权重包括本仓库dev16，不能直接用于本轮本地验证；五折成员未公开核验，也不能推定兼容。本轮只读取模型卡/元数据，没有下载该自训权重。

17:29 UTC重查本项目已失败的上下文CE与上述模型卡：旧实现只对固定span-ID做二分类（局部前后180字符、max160），随后直接阈值或固定权重与树混分，未获得收益；冠军卡描述的是折外CE分数作为二阶段树的训练特征，且有不同的候选来源。因此旧负结果只关闭已试配置，不证明所有上下文融合无效。若实施新控制，须补相应训练范围内的折外特征及候选来源，不能用训练内CE分数冒充折外，也不能为复现借用含原dev的成品权重；当前尚未启动这一新训练，先验证新候选池的实际增益。旧实验入口为[本体E-007](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/ontology-rules/solutions/ontology_rules/experiments/E-20261023-007.md)第七阶段。

[PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger)提供从本体构造远程监督样本的流程，无需额外人工表型标注。现成v1.2的HPO2023词表与比赛指定2026版本不同，且示例外部GSC开发集不能照搬；若采用，使用指定HPO与当前训练折重新构建，保留原文跨度。

已进一步审查 PhenoTagger v1.2 固定代码 `10f8ee3e` 与权重卡：代码MIT、权重Apache-2.0。梯度训练使用HPO远监督，官方示例另用GSC人工开发集选择checkpoint；代码也允许无dev固定轮次，但成品实际选模谱系未公开，不能断言每个权重都用过GSC或全为纯弱监督。旧softmax头为17,265个HPO加None，当前2026分支有2,036个ID不在旧头，不能仅替换词表。本轮从已有通用Biomed编码器与2026本体重新训练，多类encoder全量微调区别于此前冻结encoder二元MLP；先测真实耗时，不把小预算试验说成完整100epoch复现。完整源码接口/多父节点限制和来源SHA见外部 `research/phenotagger-v1.2-review/` 与来源清单。

官方参考文献中的 [AutoPCR](https://doi.org/10.1093/bioinformatics/btag304) 已读取全文与 [MIT代码说明](https://github.com/yctao7/AutoPCR/tree/c2b46a9c088c696533a43c5cb8e50c7500f055fe)：BioNER加名词短语/协调结构抽取，SapBERT置信度分流，LLM根据候选定义与同义词链接；另以HPO词面和检索难负例自监督微调linker。原方法的30B/80B LLM超过本赛上限，Stanza i2b2临床监督组件和GSC开发集选参也不能直接搬用。可借鉴坐标可追溯的协调短语分解与难负例构造；重组短语不能伪装成原文连续跨度。本轮仅研究方法，未使用其外部标注语料、结果或任务权重，尚未实现或测分；来源位于外部 `research/autopcr/`。

[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a)（Apache-2.0）作为可能的新基座做了独立可行性核查：全部775个权重键的形状推导与API总量一致，完整参数9,653,104,368，文本部分8,953,803,264；四个权重文件共19,306,310,880字节，LoRA参数仍须另计。稳定Transformers 5.20.2原生支持；普通非packed短推理有纯PyTorch回退，packed训练则需要对应FLA/conv支持。混合线性注意力要求重新选择LoRA层，不能直接复用旧8B训练器。先在自有隔离环境测短输入/单步成本，长上下文显存、吞吐和质量均未知；A800项目盘实际可用约9.45GiB，当前不能放下新权重。.100/.3有空间。源码、版本与磁盘证据见外部 `research/qwen35-feasibility/independent-review/`；这项审查没有运行模型。

## 其他同赛开源记录

`irryxeek/patientphex-chip2026` 的多来源跨度、SapBERT不同概念margin、上下文关系融合及正负报告值得研究；`yuanpinz/chip2026-PatientPheX` 给出逐occurrence、逐患者与联合患者判断、结构范围筛选；`AnXin-HITSZ/...` 与 `xi-ix/HPO` 有词典未覆盖片段的语义候选做法。其自报分数未独立复现，训练划分和评测不同，不能直接排名或沿用阈值。许可未明确的实现只参考思想；没有引入他们的标注或预测。

`wangzgui/CHIP2026` 虽为MIT且README描述完整架构，但固定commit `dfec694b2cb85e946148de90612467b8e78ee0c7` 只有README/LICENSE/gitignore，不是现成可跑实现。方法名称、公开下载和环境安装都不替代本项目实际消融结果。

## 原论文表格：可取得，但首个冻结政策没有增益

复读官方README §参赛规则3：允许公开外部资源，禁止外部人工标注表型数据集。据此，本轮可研究原PMC论文的未标注表格作为Task2外部证据，保留出处并在技术报告披露。Task1仍只输出给定全文内可核对的坐标，不把外部表格位置拼入输入。

按冻结train48中出现TABLE的文献顺序选前两篇，未依照gold漏例选取：[PMC11203018](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11203018/fullTextXML)、[PMC11235872](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11235872/fullTextXML)。均HTTP200、CC-BY，XML完整SHA见来源清单；前者第二表区分历史病例与“This Case”列，后者含13个家系成员列及认知/精神症状等行，而给定TABLE输入仅有caption。当时只确认存在额外原始证据，尚未跑分或证明收益；后续冻结对照见下。后续应按明确患者称谓对应列，处理No/Normal/unknown、脚注和历史病例，不把整表分给所有患者。

随后独立审查固定fold0的全部7篇含TABLE输入，7篇原文全部取得，15张表共552,366字节XML；保存8处rowspan、17处colspan和7组脚注，9项解析/ID边界检查通过。完整临床编号规范化能对应7个临床表患者和9个酶/基因型表的Subject，后者不能直接视为表型证据；全程未读gold、未生成预测。来源、许可、SHA及结构化缓存位于仓库外 `research/pmc-tables-fold0/`，结果manifest SHA `afeacad97fb8cc8d95a35a08469514008a14c377bc8caf5db43f0a83d80f5e75`。先在train48定义并冻结映射规则，再作fold0对照。

后续CRF路线按train48前两篇定义并冻结严格患者身份、临床表标题、精确HPO/训练别名与明确肯定单元格规则；固定fold0对照中新增31/29条关系全部FP，未获收益。外部临床事实不等于本赛标注范围，见[H25](../../HANDOFF.MD#h25外部原论文的临床事实未必属于比赛标注范围)；后续优先给定文本，不继续围绕本折扫描表格阈值。
