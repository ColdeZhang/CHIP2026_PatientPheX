# Sap64 实际 B 推断执行记录

主线程/root 已将 `.3` 的物理 GPU0、UUID `GPU-07726469-bf7a-b832-9dee-019dc276dc1d` 本阶段唯一执行权委派给本子代理。只加载既有 SapBERT、42575 bank 与 detector；保持原编码/排序/精度/阈值，不训练、不重编码 bank、不读取 dev gold、不操作官方平台。后续 GLiNER 查询须在本 B 阶段结束后顺序执行。

冻结源码与计划：scorer `8930e7cd`、driver `cd7d5bea`、[实际计划](sap-actual-run-plan.json) `fe9b4461`；父线程已在 `60754c32608156cfa28da43cd60b49b53f840caf` 提交推送并核验远端相同。预算为完整旧 dev 缓存 CPU180+10秒，成功后唯一 B100 GPU600+30秒。driver 绑定源/计划SHA、独占启动锁、自身进程组 TERM/KILL 和实际终态；B前再次检查卡号、UUID、空闲及磁盘。

远端工作目录：`sequence-round2/official-901-B/sap64`，mount `1593f159-19bc-41ca-8b4c-2e7455f5db99`。本地完整执行收据目录：`/workspace/patientphex-runs/round2-20261023/research/official-901-B/old-pool-cpu/actual-run/`。旧 `semantic-neural/model`、`ontology-round2/dev16-sapbert/embeddings` 与 detector 均只读。

06:37:42Z 实查 GPU0 4MiB/无计算进程、UUID正确、磁盘可用189.2GB，新目录此前不存在。9件部署包3,199,902B，SHA `c8ea76e58351de0ad5f981fa5ed436c695d29c826d1afff71eed3d1d47849600`；66个49152B以内分片经 MCP 上传，逐次保留响应SHA。UNKNOWN须先盘点，不重复start；LUNA故障按≥180秒退避。

当前：分片部署中，尚未 CPU/NN。完成后更新实际 terminal、GPU释放、全部100篇 scored 候选与 CPU oldpool 合成指纹；不得用准备状态冒充推断完成。独立审查正在依据已冻结源/计划及后续实际收据做增量核验。
