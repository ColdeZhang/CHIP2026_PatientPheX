# 研究笔记

- 关键区别是对患者归属连接的监督，不是 HPO 短语复制。已有 G 的 lexical grounding、DocRE latent mention、直接 SFT、DPO 都不能被称为这个教师理由任务的正负终点。
- STaR 的“最终答案正确”是监督过滤，不是忠实性证明；标签条件越强，越应防止解释只复述给定答案。
- Distilling Step-by-Step 的成熟实现支持标签与解释分任务，推断可只走标签；不要求比赛时长思考。这里是借用结构，不复刻其模型、数据规模或论文效果。
- 文本可定位、关系金标一致、归属推理成立是三个不同门。nearest、same sentence、same passage只属于位置统计。
- 研究范围：train32原金标/既有来源、公共论文与作者代码；未访问新held gold、模型或评测。root另查Sap监督bi-encoder，本线不重复。
