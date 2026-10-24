# 最高分901官方B：三public与Task1池

本目录记录本体owner的固定模型B输入推断。完整输入、raw与预测在R/research/official-best-B-20261024/ontology-three-public，不进入Git。只使用根合同2c169066的B纯输入；模型与native阈值/版本保持原accepted方法。CPU census源从已通过的BioBERT dev16版本窄改，所有NN代码删除。后继公共模型缓存不读取gold/质量，复用仅同模型同token tuple的原前向。

共同基础池由sequence构造，保留Bioformer阶段中间池；最终103特征由retrieval固定LGB803b transform/predict，不fit。官方提交由root统一执行。
