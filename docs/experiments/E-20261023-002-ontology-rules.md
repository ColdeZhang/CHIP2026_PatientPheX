# E-20261023-002：本体词表与患者状态机首版

状态：**已完成本地验证和 B 候选，建议保留**；是否提交由主线程统一决定。本路线没有操作平台。

分支 `day1/ontology-rules`，起点 `ac8a752`，实际方案代码提交 `2cf6fee`；共同评测/校验/数据基础来自 `90d52d0`，已 cherry-pick 到本分支，方案代码与已生成预测均未变。代码 SHA-256：`56cdcb45dffd821bfa7fce6ac54b28364624ae7b603058487ecb3485c5878f7a`。运行环境 Python 3.12.14，仅标准库，CPU 单线程，无 GPU/外部推理服务。

## 假设与成功标准

指定 HPO 名称/EXACT 同义词加训练折别名可建立稳定实体基线；作用域否定与患者指称状态机可减少无差别归属。先要求无泄漏、合法输出和非空有效预测，再按共同验证四项指标比较；不预设官方达标分数。

## 调研与实际方法

2026-10-23 实际读取：

- [ClinPhen README](https://github.com/kingmanzhang/clinphen)：可使用自定义 HPO 同义词表解析临床文本；其 README 说明学术、非营利和个人使用许可。成熟词表方法值得作为参照，但默认资源不能直接满足指定 HPO 版本和多患者/字符跨度输出。
- [negspacy / NegEx](https://github.com/jenojp/negspacy)：前置否定、后置否定、伪否定和终止触发词四类作用域。采用思想，未复制其代码或默认词库。

实际实现为 [run.py](../../solutions/ontology_rules/run.py)：

1. 解析官方 HPO 2026-06-23，只保留 HP:0000118 分支的 19,120 个非废弃概念及其 alt_id。
2. HPO 名称/EXACT 同义词建立 token trie；训练折标注补充别名，以频次选择歧义映射；高频出现却从未标注的非训练别名作为噪声过滤。
3. 完整词边界、简单单复数归一化、优先最长跨度；短缩写要求原文大写。前/后置否定、伪否定和转折分句控制作用域，不输出否定实体。
4. 指定患者显式 mention 优先；独有称谓在全文匹配；句内最近指称、段内患者和跨段叙述状态依次归属。非指定家属的病情不自动归于主患者。
5. 每篇文献和每位指定患者都保留；所有坐标由 passage offset 加局部切片产生，HPO ID 只来自允许分支。

没有加载归档代码、词典或预测，没有使用外部人工标注表型数据。验证输入的 entities/association 不参与算法。训练和推理文献 ID 若相交，入口直接报错。

## 数据、验证与命令

数据来自主线程重新下载并冻结的官方文件。固定划分 seed=20261023：64 篇训练、16 篇验证，验证 36 位患者。B 阶段另用完整 80 篇重新构建词表。

- HPO SHA-256：`a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b`。
- train64 SHA-256：`cb45e7df6f449f0cb9e3676e7a7eb7530a89a7541a0adae5fa0c77bfae3865f2`；dev.input SHA-256：`c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e`。
- train80 SHA-256：`3880983ee40428399f579478029342aa46678b36e4021d20494f25d8873dcdbd`；B.input SHA-256：`878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4`。
- 共享评测器 SHA-256：`588da322d72babe3ace4afc108f0b9665bc6902e1f4820e45d21a069a0806489`；校验器 SHA-256：`b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f`。

实际入口与参数：

```bash
python solutions/ontology_rules/run.py --train /workspace/patientphex-data/day1/train64.jsonl --input /workspace/patientphex-data/day1/dev16.input.jsonl --hpo /workspace/patientphex-data/hp.obo --output /workspace/patientphex-runs/day1-20261023/ontology-rules/dev-v1/predictions.jsonl --work-dir /workspace/patientphex-runs/day1-20261023/ontology-rules/dev-v1
python /workspace/CHIP2026_PatientPheX/scripts/evaluate.py --gold /workspace/patientphex-data/day1/dev16.gold.jsonl --pred /workspace/patientphex-runs/day1-20261023/ontology-rules/dev-v1/predictions.jsonl --output /workspace/patientphex-runs/day1-20261023/ontology-rules/dev-v1/metrics.json
python solutions/ontology_rules/run.py --train /workspace/patientphex-data/day1/train80.jsonl --input /workspace/patientphex-data/day1/B.input.jsonl --hpo /workspace/patientphex-data/hp.obo --output /workspace/patientphex-runs/day1-20261023/ontology-rules/B-v1/predictions.jsonl --work-dir /workspace/patientphex-runs/day1-20261023/ontology-rules/B-v1
```

两份预测均额外使用共享 `validate_predictions.py --input INPUT --pred PRED --hpo HPO --output validation.json` 校验。上述是本环境的实际路径，程序本身全部接受参数。

## 关键过程与本地结果

合成检查覆盖 `apoptosis` 不误匹配 `ptosis`、句内否定/转折、同段两患者分别归属及坐标切片一致。先用 B 的两篇去标签输入做无训练词表 smoke，输出 25 实体/4 关联且校验通过；没有评分意义。之后固定首版代码，依次运行 dev-v1 和 B-v1，没有根据验证答案修改规则。

| 指标 | dev-v1 |
|---|---:|
| menF1 | 0.72581454 |
| docF1 | 0.73716632 |
| micF1 | 0.53435115 |
| macF1 | 0.46304951 |
| Score | **0.61509538** |

这些是 `patientphex-public-approx-v1` 的独立验证近似分，**不是官方或 B 集分数**。默认 scope=all、无 ID 文档概念使用原文；正文启发式下总分 0.61407534，其他无 ID 约定下约 0.61566–0.61681。

- 验证：16 篇/36 患者，889 实体、339 关联；格式及 HPO 校验 **0 错误、0 警告**。
- 验证实体 TP/FP/FN=724/165/382，P=0.8144、R=0.6546；关联 TP/FP/FN=210/129/237，P=0.6195、R=0.4698。
- 关联漏报 237 中，132 是概念未被实体层识别，105 是概念已识别但未归到目标患者；129 个关联误报中，17 个属于文内其他患者，112 个不属于任何金标患者。这是评测后的诊断，没有用于重训或补规则。
- 漏检实例包括未覆盖的表达 “poor pupil reaction to light”“pallor optic discs”；误报包括过泛词面 “pigmentation”“pallor”。病例归属仍受跨段指代和一般讨论混淆影响。
- 训练折词表 41,834 个归一化词面，其中 2,279 个训练别名；14 类训练标注 ID 不在指定本体有效分支/格式中，已在 manifest 逐项计数并排除，不私自换新版本体。
- dev 拟合/预测 0.804/0.105 秒；B 拟合/预测 1.183/0.766 秒。为进程内墙钟时间；峰值内存未测量。

## B 候选及交接

B-v1 覆盖 **100 篇、244 位指定患者**，共 **5,812 实体、2,287 关联**；格式/HPO 校验零错误零警告。没有金标，不报告 B 本地分。

- 预测文件：`/workspace/patientphex-runs/day1-20261023/ontology-rules/B-v1/predictions.jsonl`，757641 bytes。
- 预测 SHA-256：`2c4f1e7ebd58da648cf8565687345af8f95f5f514a09cf3fe5d9f0813c24e7c7`。
- 验证预测 SHA-256：`5a994261965f7887f2bfef384401380d49581940435bbe46d19604f6e4868e4b`。
- 两个运行目录均保存 `manifest.json` 和 `validation.json`；dev 另有完整 `metrics.json`、少量例子的 `error_analysis.json`。大预测只保存在仓库外，可用已记录入口重建。

结论：首版不是明显失败路线，建议保留为可解释参照并交主线程决定提交。主要改进空间为新表述召回及患者关联，而不是继续堆缩写硬编码。当前没有平台提交 ID/官方分；官方结果按用户要求后续再回收。
