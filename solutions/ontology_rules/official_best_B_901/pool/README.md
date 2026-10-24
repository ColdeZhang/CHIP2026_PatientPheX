# 固定 Task1 候选池

目的：把原 train64 规则、Sap64、CRF 的 B 结果与三套公共 PhenoTagger 缓存按已接受方案组合，交给 retrieval owner 的固定 LGB803b/103列 vectorizer；本入口不加载、拟合或调用模型，不解码实体、不评分。

`build_pool.py` 从六个固定 SHA 源文件只编译指定纯函数：原 `single.pool`、`multi.pool`、PubMed `merge_pool`、Bioformer `merge_bioformer`、BioBERT `merge` 及官方 HPO parser。底座顺序与原 `(offset,length,identifier)` 精确键不变，公共候选沿 native 首次出现顺序追加。PubMed/Bioformer 的原 head 元数据保留，BioBERT 同键分数取实际 hybrid max。坐标、当前 HPO 分支、全部输入 ID、native provenance 与实际权重身份均检查。

**CRF confidence 不进入本池。** 接受方案原 reference51c5 的 `crf_confidence` 为 null；新源拒绝带 `crf_ner/crf_link` 的 CRF 实体。原 feature 函数继续默认0。三来源各加6字段，原85列加18列的实际 vectorizer transform 由 retrieval 执行。

已完成唯一成功 dev 重建，1.862476秒/rc0，无 gold：

- 候选数 12,850 → PubMed 13,063 → Bioformer 13,144 → BioBERT 13,246。
- 最终 `input.pool.jsonl.gz` 的压缩字节及解压字节均与既有18ac完全一致。
- Bioformer中间池与原c032候选去掉 `selector_probability` 后的值、类型、字段顺序逐字节一致。
- 两个旧public prepared文件中的dev行逐字节一致。

见 `dev-reconstruction-result.json` 与 `source-manifest.json`。第一次入口在输出前发现旧Bioformer manifest未包含可选 `model_type`；保存原source/plan/stderr后，改为有type时检查，并始终用完整H5 SHA锁模型身份。该接口纠正不改候选语义，不重跑模型或评分。

B源尚未执行。`B-plan.template.json` 的 `execution_ready=false`，还缺完整 Sap scored-candidate ref（不能填raw候选）及三套 native final manifest/canonical/provenance。owner按准确B资产填完、核原train64来源并固定新plan之后运行：

```bash
timeout --signal=TERM --kill-after=5s 120s python build_pool.py --plan /absolute/path/B-plan.json
```

输出使用新目录，已有目录直接拒绝；保留 `base.pool`、`pubmed.pool`、`bioformer.pool`、最终 `input.pool` 四阶段与两public prepared文件。全文池和预测全部留仓库外：`R/research/official-best-B-20261024/ontology-three-public/pool/`。本目录仅源、配置及摘要。原dev复现产物不可当B输出。
