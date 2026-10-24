# 完整双采样后的必要 Sap 查询

此目录只准备运行入口，尚未绑定真实 held 输出或启动远端查询。固定 `.100 GPU0`（`GPU-126293b9-6677-8464-cee7-01e693efbf29`），须先收到原双采样完整 78 条及该卡实际释放证据，再通过查询 driver 的 fresh UUID、显存、进程和磁盘门。唯一查询时限为 180 秒，TERM 后最多 30 秒；不与原 9B 模型共驻。

输入直接使用 `postprocess/process.py prepare` 产生的 `grounded-sampled-normalization-preparation-v1`：`extraction` 字段、两个 seed 各 39 条、train32 来源、`queries=[{"quote": ...}]` 与 `necessary_public_query_count`。两个流的完全相同词面只查询一次。binder 不读取金标，不创建假输入或假完成记录；必要查询为 0 时不生成 GPU plan。

复用原 `strict.py`、`semantic.py`、`public_bank.py`、`policy_v1.json`、`held_query.py`、`held_driver.py` 的准确 SHA。公共 41099 行 bank、labels、manifest 与 SapBERT 六件文件保持原字节，train32 aliases 仍由已有 postprocess 在本地按固定来源恢复。查询阶段不加载 train64 aliases、不拟合、不重编码 bank。

执行目录必须保持独占。以下命令仅在真实完整采样及 prepare 到齐后由 owner 顺序执行；当前未执行：

```bash
python solutions/grounded_sampled_consensus/normalization_runtime/bind.py \
  --preparation-manifest /workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1/normalization-preparation-v1/manifest.json

python solutions/grounded_sampled_consensus/normalization_runtime/remote.py pack \
  --plan solutions/grounded_sampled_consensus/normalization_runtime/held-query-plan.json \
  --directory /workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1/normalization-runtime-v1
python solutions/grounded_sampled_consensus/normalization_runtime/remote.py deploy \
  --directory /workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1/normalization-runtime-v1
python solutions/grounded_sampled_consensus/normalization_runtime/remote.py start \
  --directory /workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1/normalization-runtime-v1
python solutions/grounded_sampled_consensus/normalization_runtime/remote.py observe \
  --directory /workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1/normalization-runtime-v1
python solutions/grounded_sampled_consensus/normalization_runtime/collect.py \
  --directory /workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1/normalization-runtime-v1

python solutions/grounded_sampled_consensus/postprocess/bind.py finalize \
  --query-collection /workspace/patientphex-runs/round2-20261023/research/grounded-sampled-consensus-v1/normalization-runtime-v1/collection.json
python solutions/grounded_sampled_consensus/postprocess/process.py finalize \
  --plan solutions/grounded_sampled_consensus/postprocess/finalize-plan.json
```

若必要查询数为 0，跳过整个 remote/collect 阶段，调用原 postprocess `bind.py finalize` 时不传 `--query-collection`。不因词面拒绝或解析失败而重采样。

`remote.py` 与 `collect.py` 相对原 full64 工具各只替换一个私有 namespace。原 claim、唯一 start intent、UNKNOWN 库存恢复、退避、分块 SHA/fsync、原完整终态收件逻辑保持。UNKNOWN 不能解释为未启动，不能换目录重开同一任务。collector 保留全部 query scores/runtime/终态，向量保留在远端准确 SHA locator；返回 `grounded-held-query-complete-collection-v1`，可直接交原 postprocess finalize。

当前 `.100` 的实际环境是 Torch 2.7.1+cu126、Transformers 4.51.3、tokenizers 0.21.2；原 `.3` 是 Torch 2.7.0+cu128、Transformers 4.50.0。同资产与参数已证，未声称不同运行时逐位等价。每次真实查询会保存 encoder runtime 并核原参数、CLS_L2、max_length32。

`metadata_checks.py` 仅检查合成元数据接受/拒绝、已有小收据与源 SHA；不会建立实际 query plan，也不会访问 MCP、CUDA、模型、bank 正文、训练标签、held gold 或评分。
