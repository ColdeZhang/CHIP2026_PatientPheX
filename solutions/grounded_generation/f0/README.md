# 同模型 F0 固定复验

目标：同一 d52e2e6c 生成模型＋b59eb 固定规范化，在另一内部留出折验证强支持完整组合的增量；同时给主线程已锁定的三源 2of3 提供完整新关系。不因本线分数负删除共识源。

- 训练和经验 aliases 固定 F2+F3 train32；F0 16 篇47患者全部保留，全文无截断。公共41099 bank纯化来源沿 dac8 独审，不将旧42575 bank 的经验标签当公共知识。F0反复用于开发，不冒称盲测。
- 本地47输入289420 tokens/max9935；A800实际native47 CPU5.522秒通过，11件完整collection5bcd8c19。A100 Torch2.6.0→A8002.7.1如实披露；TF5.20.2、原native backend、发布tokenizer、greedy1600、原decoder/解析/规范化不变，无FLA。
- 173MB inference-only cache仅adapter/config/README/checkpoint，原完整optimizer/RNG与两durable ACK仍A100。copy前后四SHA/fsync/reopen。不能把A800缓存称为完整训练checkpoint。
- 主线程授权实际CPU/cache通过后唯一47fresh **2100+TERM30**。已有F1全39不重跑。原意图存在先恢复，UNKNOWN不重启；没有自动尾行或提高cap授权。
- 完整raw47→原strict quote解析→固定b59eb；必要Sap查询一次≤180+30，.3真实空闲才执行，不重bank/fit。
- 唯一质量：强F0 B（94504b97，Score .7115075800266846）∪全部新关系；保947实体/全部原关系和非关系字节，一完整预测冻结后1v2/旧0。严格总Score上升才保独立跨折候选，micro/macro并列。new-only只输出不评分。无原dev、官网、新训练。

入口：`cpu_remote.py`/`cpu_collect.py`；`copy_remote.py`只原字节传输；`bind.py`在actual完成后绑定不可变`infer-plan.json`；`remote.py` pack/deploy/start/observe；`watch.py`只读180秒；`collect.py`完整原raw回收。后继`extract_quotes.py`/`held_prepare.py`/固定Sap/`held_finalize.py`，`actual.py`原F1链机械47适配，`quality_control.py`固定唯一强union。

当前：CPU47与四件缓存完整PASS，原47唯一11:38:58.166 A800启动proc1a64fada，TERM12:13:58/KILL12:14:28；actual-start.json提供真实metadata与只读watcher入口，不能重复start。此前LUNA与初始2GiB保守盘门失败保留，已按实际173MB/cache＋raw预算调整768MiB/512MiB并经root确认。
