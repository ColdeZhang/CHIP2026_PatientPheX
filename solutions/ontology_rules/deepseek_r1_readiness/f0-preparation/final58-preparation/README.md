# 条件式 F0 完整58准备

**条件未通过，已关闭且从未启动**：F1 union .634402907 <旧 .660942003，macro亦下降；[关闭记录](closed-before-start.json)。原pending与决定文件保持不变，无执行plan.json、无新GPU/质量。

新增后继决定96b30159于F1新质量前冻结，不回写F1原primary/secondary/automatic_migration=false。只有完整且核验的F1 secondary-union满足原retain（mic+mac严格升、mac不降），才允许F0 6000+TERM30／idx30精确缓存1＋57fresh；否则关闭本扩展。F0固定两endpoint共同冻结后各唯一v2，旧SFT48指标复用，不加第三组合。

真实标准库完整58输入/cache合同，本地0.044110秒、A8000.077232秒均PASS，195157输入tokens，原idx30全3427IDs及plan/source/wrapper/模型参数/原raw绑定。CPU仅在内存将pending plan的execution flag置true调用同一纯load_inputs；无launcher、native tokenizer、model、gold/评测，磁盘计划一直execution_ready=false。

173140B source/两输入包已实际部署到A800自有f0-full58-v1；原copy资产及最长缓存原位复用，不重复权重/推断。实际项目盘余151781376B。新增launcher必须读F1 PASS gate证据，SHA/大小和union原retain及独立终点通过字段都成立才可进入原freshGPU gate；其他模型、采样、codec源不变。

当前plan.pending e505ba5d仍禁用、gate PENDING；无完整58GPU启动。F1原唯一job与自动收集/两评测保持不动。准备清单505a6974，实际远端输入回执见actual-input-contract.json。
