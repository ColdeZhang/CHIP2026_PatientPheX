# BioBERT固定原dev无标签缓存

root授权同既定第三head的唯一原dev开发迁移。只用固定c0fc四字段输入、原2–8/native五位>.95/dict/abbr/currentHPO，与train64完全相同。仅复用同BioBERT真正lower-token tuple原始top3；不借用另外head概率，不读取gold。

实际CPU20.483秒：16文献/514段/44998token坐标/64288候选/55203unique，4655实际缓存键可复用，50548需新forward，词典979。基于刚完成同head的真实吞吐和全fixed开销估计107.54秒；owner冻结更严格native180+TERM30（CPU源parent名义300保持历史）、来源CPU120。源/预算先推送；必须root明确分配及freshGPU2空闲后才能运行，没有逐步质量授权要求。

恢复入口：外存 research/public-phenotagger-biobert-dev16-cache；A100项目 public-phenotagger-biobert/dev16-cache；prepared与train64 raw同机，权重不用重新下载。完整cache+来源合同通过后交full64 HGB唯一迁移，当前未GPU/质量。
