# 来源与假设

已读 main `bc8084e7` 的 AGENTS、HANDOFF H59、WORKLOG 与方法地图，并核实际源码/旧端点摘要。实体删除本身已试：旧 Qwen3 canonical NO veto 删除133=60TP73FP，Score .728697536469→.722573801595；旧9B binary veto删除152=71TP81FP，.728144787300→.720623998953。两次都是负结果，不能称本次首次实体否决。

本次唯一新信号：保留原2931 HPO/NIL/NO目标，再增加train32的539严格crossing边界目标，BOUNDARY与NO独立，fresh两轮868更新。完整跨度本身可能是合法嵌套或复合；只有新模型对**同一exact span**输出BOUNDARY才能删除。假设是该边界信号可能在原强选择器中减少更多错误跨度，同时损失较少正确跨度。只能由预锁完整控制证实，不能据训练成功、父union或new-only的好坏推断。

父 owner 已确认未来真实字段，当前已核 prepared manifest/held/encoded 原件。1416 row_id、1416 `(pmc,offset,length,quote)` 全唯一；强base1313实体全有唯一匹配。当前base无复合ID/同span多对象，合成合同仍覆盖这些合法情况并统一处理。溯源 SHA 见 [source-provenance.json](evidence/source-provenance.json)。没有读取父 held 模型结果、真正 gold 或质量；没有远端观察。
