# 第二轮研究索引：直接指导可检验的改进

本页由主线程维护，只保留共同方法依据；具体实现、实验过程和取舍由三方案各自记录。下载的论文/原文在仓库外，[来源与 SHA](round2-research-sources.json)可用于核验，不重复查站点。

## 主办方原始证据

已下载并读取[官方评测综述](https://arxiv.org/pdf/2610.29606) 3.3、4.2–4.3 节；HTML 路由未成功，PDF可用。论文前三名 B Score .7389/.7247/.7169 是官方不同测试集结果，不当作本地目标或可复现保证。

| 来源 | 可实施的思想 | 本轮落点 |
|---|---|---|
| 官方基线 §3.3 | Qwen3-8B 先扩充指定患者称谓/共指，再判断每批8个表型的归属，局部上下文前后各500字符；greedy，输出上限1800/2600 | 检索路线先做既有候选关系对照，再做原文quote补充 |
| Agriv §4.2 | 多个PhenoTagger+监督跨度+HPO候选，SapBERT检索，两阶段LightGBM与上下文交叉编码器；LLM证据及位置特征经OOF融合 | 本体与CRF补候选/标准化，检索研究可学习关系判断 |
| 第二名 §4.2 | 多源候选投票、LLM处理重叠/否定；生成患者HPO集合与候选对分类的互补SFT | 检索零/少样本之后再决定是否折内LoRA，避免只扫阈值 |
| 第三名 §4.2 | 监督跨度、SapBERT、本体图特征；关系模型及LLM assertion复核，保留关系分数用于恢复 | 实体与关系使用可追溯候选池，避免最终实体阈值硬截关系 |

冠军公开[权重卡](https://huggingface.co/rigidk/CHIP2026-PatientPheX-Agriv) revision `7a98ca95683eb920b09cc9141cdf512dc5f1fa41` 声明所有自训标签来自官方80篇。可研究结构（BioLinkBERT-large、Qwen3-8B LoRA r16、单/多患者LGBM），但这些已训练权重包括本仓库dev16，不能直接用于本轮本地验证；五折成员未公开核验，也不能推定兼容。本轮只读取模型卡/元数据，没有下载该自训权重。

[PhenoTagger](https://github.com/ncbi-nlp/PhenoTagger)提供从本体构造远程监督样本的流程，无需额外人工表型标注。现成v1.2的HPO2023词表与比赛指定2026版本不同，且示例外部GSC开发集不能照搬；若采用，使用指定HPO与当前训练折重新构建，保留原文跨度。

## 其他同赛开源记录

`irryxeek/patientphex-chip2026` 的多来源跨度、SapBERT不同概念margin、上下文关系融合及正负报告值得研究；`yuanpinz/chip2026-PatientPheX` 给出逐occurrence、逐患者与联合患者判断、结构范围筛选；`AnXin-HITSZ/...` 与 `xi-ix/HPO` 有词典未覆盖片段的语义候选做法。其自报分数未独立复现，训练划分和评测不同，不能直接排名或沿用阈值。许可未明确的实现只参考思想；没有引入他们的标注或预测。

`wangzgui/CHIP2026` 虽为MIT且README描述完整架构，但固定commit `dfec694b2cb85e946148de90612467b8e78ee0c7` 只有README/LICENSE/gitignore，不是现成可跑实现。方法名称、公开下载和环境安装都不替代本项目实际消融结果。

## 原论文表格：仅完成可取得性研究

复读官方README §参赛规则3：允许公开外部资源，禁止外部人工标注表型数据集。据此，本轮可研究原PMC论文的未标注表格作为Task2外部证据，保留出处并在技术报告披露。Task1仍只输出给定全文内可核对的坐标，不把外部表格位置拼入输入。

按冻结train48中出现TABLE的文献顺序选前两篇，未依照gold漏例选取：[PMC11203018](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11203018/fullTextXML)、[PMC11235872](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11235872/fullTextXML)。均HTTP200、CC-BY，XML完整SHA见来源清单；前者第二表区分历史病例与“This Case”列，后者含13个家系成员列及认知/精神症状等行，而给定TABLE输入仅有caption。只确认存在额外原始证据，**尚未跑分或证明收益**。后续应按明确患者称谓对应列，处理No/Normal/unknown、脚注和历史病例，不把整表分给所有患者。
