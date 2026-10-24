# BioBERT固定原dev开发迁移

在train64四fold固定控制满足keep且独审2907fc55通过后，主线程授权唯一迁移。原full64 HGB训练4个OOF来源与现有4held池的base/candidate/CRF/input SHA逐项相等，复用原61781候选及已核6来源字段/new500共62281；原dev只读无标签输入，native缓存由单独head产生。

先独立CPU绑定train64来源，再同HGB160拟合并冻结真实model/vectorizer。后续固定.3/proper-nested、强模板d531（.737585917）的全部非entities/368关系原样；完整新pred先freeze后唯一v2，不重算关系支持池、不扫cut。原dev已用于开发，不称盲测。

资源：训练准备30秒、fit60秒/单线程；新native缓存由单独source/plan控制，A100租约必须明确。训练侧计数和成本依据已完成四fold55.703秒、原full64旧fit13.719秒，不把估计当实测。新增head后共享生产库存9,597,467,990参数，实际adapter另按既有库存计算，同Qwen权重仅计一次，关闭的其他模型不加入。
