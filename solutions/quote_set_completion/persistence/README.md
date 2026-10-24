# QSC F1 最小补包

此入口只保存已冻结的 F1：entry `a9044674`，review `b3656560`，审查 manifest `2f2f753e`。源码严格取 entry 的 `source_refs` 和 `refs`，不遍历当前源码树；新增 F0 不进入补包。审查的安全快照按其 manifest 精确复制，不递归打开依赖。

原 CPU15、失败10、partial41、两形状23、formal34、infer92、query14 的终点 ZIP 均逐字节保留，附原 collection/index/package。失败和 partial 不改称成功。补包另包含完整 F1 抽取、规范化、质量，以及全部39条 prompt/audit/known/encoded 原文件；不重新解析、规范化或评分。

新模型和完整 optimizer/RNG 只引用 `.100` 的 epoch001 持久 ACK `4b486327` 及其逐文件 SHA。A800 SHM 原副本不冒充持久存储；旧 d52、公共模型与 bank 不重复复制。

本地输出：`/workspace/patientphex-runs/round2-20261023/research/quote-set-completion-v1/endpoint-F1-supplement-v1/`。目标 mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，相对目录 `research/quote-set-completion-v2/F1-supplement-v1`。

```bash
python -B solutions/quote_set_completion/persistence/F1_supplement.py inventory
python -B solutions/quote_set_completion/persistence/F1_supplement.py pack
# 本次仅完成本地包；部署由主线程协调窗口后执行：
python -B solutions/quote_set_completion/persistence/F1_supplement.py deploy
```

`pack` 只允许一次，原包存在即拒绝重打包。部署复用已核 sampled/sequence 同字节分块协议，UNKNOWN 先核原库存续缺块，不重包或重模型。恢复下载 whole ZIP 与外部 manifest，核 whole SHA、逐成员 SHA/fsync/reopen、嵌入 manifest 字节一致。ZIP 总成员数始终为内容成员数加 1；本地 build 不代表服务器部署或实际下载恢复完成。
