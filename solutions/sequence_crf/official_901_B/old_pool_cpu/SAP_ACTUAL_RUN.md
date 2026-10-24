# Sap64 实际 B 推断执行记录

主线程/root 已将 `.3` 的物理 GPU0、UUID `GPU-07726469-bf7a-b832-9dee-019dc276dc1d` 本阶段唯一执行权委派给本子代理。只加载既有 SapBERT、42575 bank 与 detector；保持原编码/排序/精度/阈值，不训练、不重编码 bank、不读取 dev gold、不操作官方平台。后续 GLiNER 查询须在本 B 阶段结束后顺序执行。

冻结源码与计划：scorer `8930e7cd`、driver `cd7d5bea`、[实际计划](sap-actual-run-plan.json) `fe9b4461`；父线程已在 `60754c32608156cfa28da43cd60b49b53f840caf` 提交推送并核验远端相同。预算为完整旧 dev 缓存 CPU180+10秒，成功后唯一 B100 GPU600+30秒。driver 绑定源/计划SHA、独占启动锁、自身进程组 TERM/KILL 和实际终态；B前再次检查卡号、UUID、空闲及磁盘。

远端工作目录：`sequence-round2/official-901-B/sap64`，mount `1593f159-19bc-41ca-8b4c-2e7455f5db99`。本地完整执行收据目录：`/workspace/patientphex-runs/round2-20261023/research/official-901-B/old-pool-cpu/actual-run/`。旧 `semantic-neural/model`、`ontology-round2/dev16-sapbert/embeddings` 与 detector 均只读。

06:37:42Z 实查 GPU0 4MiB/无计算进程、UUID正确、磁盘可用189.2GB，新目录此前不存在。9件部署包3,199,902B，SHA `c8ea76e58351de0ad5f981fa5ed436c695d29c826d1afff71eed3d1d47849600`；66个49152B以内分片经 MCP 上传，逐次保留响应SHA。UNKNOWN须先盘点，不重复start；LUNA故障按≥180秒退避。

部署9件全部SHA/AST通过。CPU完整16篇已06:43:09Z以9.746秒rc0结束，proc `64bb7598`，12811候选全部旧向量缓存；原字段/ID全等、概率差0、相似度最大差1.5497207641601562e-6、两阈值决策零差。不是分数全字节相同。manifest `28e2287f` 已本地回收并获统一增量审查通过。

B100已06:44:26Z以19.219秒rc0结束，proc `58e193a2`，90303候选、100篇全fresh编码、stderr0。manifest `b4e2f2a6`，driver前后均核GPU0 4MiB/noapps。远端100篇score与embedding逐SHA并fsync通过；完整回收zip3,948,353B SHA `ee68dabee61e4be54fb690460cc3605b6860e8c3d60ef4d44a865022d9628754`。

GLiNER48的7048真实B候选已纯CPU准备为4671unique/100小batch，原metadata与ontology_exact保留。新固定append+loaded corrected LR已原dev `c1c2e508` 完整byteexact，955实体/534关联/新增32一致。后继同卡180+30计划 `effde0fb` 与driver `4732a9c7`、CPU准备/合并源 `71cfb0e9` 已父线程 `4aed0eb` 推送核验。

传输恢复已完成：GLiNER部署前只读inventory曾返回UNKNOWN，后续库存后仅续传缺片。主B原包收至2,097,152B后明确LUNA_OFFLINE，06:48:25起185秒退避；后续小段读取仍偶发UNKNOWN。按root授权，以仅允许指定客户端、单一文件、Bearer验证和180秒TTL的私有HTTP将原zip从 `.3` 只读传到 `.100`，原包大小及SHA相同，再续取原字节到本地；未重打包/重算。HTTP自身进程已停止且临时令牌已删除。`actual-run/luna-offline-backoff.json`、各失败原响应、`private-http-fetch-100.json`、`private-http-cleanup.json` 均保留。主B253件完整逐SHA通过，90303原字段逐项及100篇序列化SHA核验后已直发本体路线。

GLiNER Sap唯一实际作业 `proc_288b5e0e-477e-42c9-a073-a34689ca07bc` 在06:58:31Z启动，10.100秒rc0结束，100篇/7048候选全fresh，输出manifest `92369a9a`，GPU0终态4MiB/noapps。原scorer未变；211件包534719B、SHA `a593cc77bdde347f667e1142f9bb340e6ac945412a888792b7828c1e2eff055a` 全部本地校验。完整gzip SHA `1db125997eb0140a53d737cc6751eb44fa6a460da5a5cbb39a2b978edbaf6562`，原候选字段和100篇序列化SHA均核验。

最终 `B-full-lr/full-lr.jsonl` 为817521B、SHA `77322710caeb8d87beada5695b73d8c4c7a6f0512a7ad399b220482a6002a184`：100篇、244患者、6101实体、3415患者关联。原CRF+Sap基底5871实体全部保留，GLiNER新增230，原3324关联全部保留，新增91；保存的corrected train64 LR按原0.55阈值重算全pair，完整fallback可用。manifest SHA `6095fd6985a8f9adb10cf9e8c1e4b701e264531cfc0b0cf991fa29903cf68a01`，实际schema以 `outputs['full-lr.jsonl']` 绑定预测。已立即交付检索/父线程及统一只读审查。

结构核验通过B文档/患者顺序、全部原文offset与text、基底实体和关联保留。原源码 `association_keys` 对identifier=-1使用原文，故18个继承的未链接实体及13个原文关联按原行为保留；初始过严的仅HPO核验据源码修正，预测未修改。完整核验及收据见 [terminal-summary.json](terminal-summary.json)。本阶段没有重新fit、bank编码、devgold读取或官方操作；两次实际GPU编码均只执行一次，所有自身GPU/HTTP进程均已结束。
