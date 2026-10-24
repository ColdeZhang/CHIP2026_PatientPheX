# 文档级关系工作台账

- 起点：main `0c0672d`，分支 `explore/document-relations`，独立worktree同名；用户目标原固定dev≥.8。
- 当前实验 `E-20261024-DOCRE-001`：复现ATLOP式多提及池化与局部attention交集，扩展为覆盖全文的重叠窗口，固定公共三模型提供HPO候选；不复用任务训练候选，避免新的留出泄漏。
- 先完成F2+F3 train32→F1初版。正文全部编码；物理提及只归属一个完整窗口，跨窗重复不重复计权。患者/概念多提及logsumexp，ATLOP式pair context＋分块双线性头。首版不加额外实体图Transformer。
- 二元关联使用单logit BCE/零logit阈值；不把二元退化的ATLoss当新优势。正例和负例都按实际关系标签生成，无人工evidence标签。
- CPU准备中，尚无训练/质量。GPU由root调度，span路线先用.3 GPU0，本线在其明确释放后fresh检查。
- 恢复：外部 `/workspace/patientphex-runs/round2-20261023/stage8-document-relations/`；源码入口本目录`prepare.py`。

旧local pair CE、identity bridge、doc-HPO max-MIL都有真实负终点；本控制区别是完整文档的一次共享编码及实体全提及表示、pair共有attention，既不是局部独立片段分类，也不是取出现max分数。详见NOTES及实验协议。

准备已完成：train32候选4495对、727/818训练关系可由公共候选覆盖；held2874对。仅正文/原患者及冻结public候选进入held，未读F1 gold。CPU编码已独立部署8件/412068B并唯一启动，等待真实token/window及pair几何统计；尚无GPU。源码处于初版工程状态，CPU与GPU终点尚待。
