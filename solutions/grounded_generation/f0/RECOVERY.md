# 原 F0 唯一作业接续

先读 main HANDOFF 与本线 WORKLOG，核实际文件。当前原 A800 `proc_1a64fada-6c00-43a3-806e-b9ea4b755862` / driver63172 / child63188，11:38:58.166起，TERM12:13:58.166、KILL12:14:28.166。任何start-intent存在就禁止再start。原本地只读watch session54389，每180秒；恢复会话先核watch实际PID，不假设还活着，不并发观察同一mount。

以下在worktree根运行。`S=solutions/grounded_generation/f0`，`B=/workspace/patientphex-runs/round2-20261023/research/grounded-generation-v1`。每一步只在前一步完整实际产物通过后进行；已有冻结产物只核原件、不覆盖重跑。任何失败先保留原intent、raw、完整终态。

1. 原watch见wrapper终态后，`python $S/collect.py --directory $B/infer-F0-v1`，恢复入口自动先库存原ZIP，完整SHA收齐；不是新生成。完整47之前无质量。
2. `python $S/bind_flow.py extraction` 绑定实际47；计划及时push后 `PYTHONPATH=. python $S/extract_quotes.py --plan $S/extraction-plan.json`。原raw只解析一次，无repair/redecode。
3. `python $S/held_prepare.py --extraction-manifest $B/F0-quotes-v1/manifest.json --output-dir $B/F0-normalization-preparation`。只固定train32 aliases/public规则，产完整必要query表。
4. 若query>0：与document_relations确认其QwenF0原58完整释放；`python $S/held_bind.py --preparation-manifest $B/F0-normalization-preparation/manifest.json --release-evidence <实际release文件>`。先push实际plan；用`held_remote.py pack/deploy/start/observe`，`--directory $B/normalization-held-F0-v1`，pack另给`--plan $S/held-query-plan.json`。原180+30、同.3 Sap/public41099 bank，不重bank/fit；fresh卡/盘门。完整后`held_collect.py --directory ...`。若query=0不启动GPU并由finalize显式绑定None。
5. `python $S/bind_flow.py normalization-finalize`；计划push后 `PYTHONPATH=. python $S/held_finalize.py --plan $S/held-finalize-plan.json`。保留所有quote坐标、accepted/abstain理由。
6. `python $S/bind_flow.py pure`；计划push后 `python $S/build_pure.py --plan $S/pure-plan.json`。全47新关系输出和freeze，不评分；给root纯keys、new-only pred、normalization/actual链，不因本线最终负删source。
7. `python $S/bind_flow.py quality`；固定template与actualrefs先push；`python $S/quality_control.py assemble --plan $S/quality-plan.json`，再`score`同plan。只强B0 .7115075800266846∪全部新关系，1v2/旧0，30+1。947实体/全部原关系/非关系字节保持，macro/micro均报。无new-only v2/阈值扫描/原dev/官网。
8. 完整terminal/source/实际SHA/四分项/成本记录并push；统一终审复用0fcffd/dac8，仅新47/runtime/cache/输出/冻结顺序增量。大文件仍外存，稳定小包持久并交root唯一恢复。

A800推断缓存`P/grounded-generation-v1/F0-inference-cache/epoch-002`有4件d52 adapter/config/README/checkpoint；不含optimizer/RNG。原全训练状态仍A100 `P/grounded-generation-v1/formal-v1/output/epoch-002`，两ACK完整。F1补包184内容已root实际下载恢复c48d3776，不重复下载或模型。
