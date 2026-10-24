# 研究与边界

- 只读查重核 main 方法总图、旧 `sampled_consensus/{sampling.py,infer.py}` 与 grounded F1/F0 完整源。前者同固定双seed但输出闭集CID，presence0；后者只有greedy quote终点。本次开放quote双采样尚未发现既有实跑。
- 共享词面经同固定normalizer映射后才在 doc/patient/literal phenotype 空间求交；不同表面词相同HPO可一致，同词面不同患者仍不同关系。严格解析失败/cap整行弃答，不取合法前缀。
- 保留B4全部1189实体/39患者/524关系，不对患者、seed、阈值或TP择优。单seed仅供给/审计，不评分。
- 显存范围继承同d52/native F0 max9935和新full64 max8858真实推断；新F1原native最大10080已在A100同d52完整通过，A80080GB有充分空间。启动仍fresh检查GPU/盘/模型实际身份。

## 双开放quote采样的已实测边界

固定同模型两seed的规范化关系交集，在F1新增14条9TP5FP、F0新增13条2TP11FP。两次采样一致不代表关系正确；本配方跨折门失败后关闭，不用结果倒推新seed/阈值。截断5条和JSON失败属于完整原输出，按既定整行弃答且不重试；不能把未来改解析或候选受限的独立机制冒充本次成功。精确源/结果见experiments/GSC-20261024-001.md及evidence/F0-result.json。
