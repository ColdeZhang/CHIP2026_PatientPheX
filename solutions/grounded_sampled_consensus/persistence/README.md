# F1 完整终点补包

目标是保存已独审的开放 quote 双采样 F1 正终点，便于丢失本地环境后恢复。本目录只修改备份入口与小回执，不改变实验源、原始输出、模型或评分。当前 F1 Score 为 `.7052260831087024`；这是内部开发折结果，不是原 dev 或官方分数。

`F1_supplement.py` 固定绑定 `F1-complete-review-entry.json`（9bd386cc）、终审（c54828bd）和审查 manifest（ddd801e5）。源码来自该冻结入口的准确 allowlist；不遍历正在变化的整个方案目录，不收新的 F0/94 条准备或 `postprocess_F0`。旧 G32 中名字带 F0 的 adapter 复制祖先只作为既有模型来源保留，不能误称本次 F0 结果。

补包包含原 CPU 12 件、NN 173 件、Sap 13 件的三个原始 ZIP（逐字节保持）及 collection/index/package，全部 F1 quote 抽取、规范化、单一质量终点，实际源/计划及终审。模型、完整 optimizer/RNG、公共 bank 只保存准确 locator：d52 adapter 的 `.100` 四件推断缓存沿 ca3445a8/8f97e1dc，原 G32 完整两轮 state 沿 A100 原 collection/已恢复补包链；不复制大权重或重编码 bank。

使用顺序：先检查源码与本地只读 inventory，提交并推送源码，再打包一次。部署前由主线程交接 `.100` CPU/MCP 窗口。

```bash
python -B solutions/grounded_sampled_consensus/persistence/F1_supplement.py inventory
python -B solutions/grounded_sampled_consensus/persistence/F1_supplement.py pack
python -B solutions/grounded_sampled_consensus/persistence/F1_supplement.py deploy
```

本地固定目录：`/workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1/endpoint-F1-supplement-v1/`。

持久目标：mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，相对路径 `research/grounded-sampled-consensus-v1/F1-endpoint-supplement-20261024/`。恢复时主线程唯一下载 `endpoint-supplement.zip` 与 `members-manifest.json`，按实际 whole SHA、逐内容成员 SHA、嵌入 manifest 字节一致核验。**ZIP 总成员数 = 内容成员数 + 1 个嵌入 manifest**。

部署沿已验 `endpoint_supplement.py` 的同字节分块与逐成员 SHA/fsync/reopen 协议；UNKNOWN 时先核原上传库存，保留分块和原包，禁止重打包、重新生成或重评分。服务器完整回执通过与主线程实际下载恢复分别记录，未取得后者前不宣称已实测下载恢复。
