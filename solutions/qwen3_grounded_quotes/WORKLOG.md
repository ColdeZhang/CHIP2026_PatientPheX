# Qwen3 患者逐字 quote 工作台账

独立分支 `research/qwen3-grounded-quotes`，起点 `f960ae41`；仅维护本目录。原dev最高.7692357未改变，本线目前只源/CPU准备、无GPU/no quality。

- 检验成功G的原87 literal-quote目标迁移到Qwen3-8B是否提供互补关系；旧Q3 closedCID和canonical-name负端点保留，不冒称首次dense attention。查重与外部NLI低优先提案见NOTES。
- 预锁原G全部87目标、2轮22Adam及b59规范化；完整39后唯一 `B4 ∪ [G_Q3 ∩ (G_old ∪ S ∪ Q ∪ C)]`，F1严格正才F0。原801anchors/814atoms和4未表示gold边界不变。
- 复用已核Q3 `full64_component/native.py/common.py/loss_ops.py` 的参数化训练/状态接口；CPU复用canonical的native renderer/完整census。原messages/answers字节不变，模型模板/IDs实际检查另做。
- 现有.100公开8B基座与TF4.51.3，无大下载；资源未交卡，不发GPU请求。GPU探针/正式/39推断预算须新形状与实耗支持，不能套旧canonical峰23.617GB。
- 下一步：先push此准备检查点；唯一50+5 CPU完整126原生/HF对照→全产物回收→预算与最小shape delta计划。原GPU队列由root安排。
