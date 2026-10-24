# 闭集全文 fresh2ep：最小源码增量

本目录仅准备源码与本地 CPU 合同，未启动 GPU、MCP 正式作业、原生 tokenizer、模型或评分。root 已给定的正式边界由 owner 冻结和执行：训练两次载入、两 epoch、checkpoint 与完整持久 ACK 共用 2700 秒，加 TERM 30 秒；固定全文 80 条推断另 900 秒，加 TERM 30 秒。不得据本目录的 CPU 检查宣称训练或质量已完成。

唯一输入变量是沿已审原文来源向原消息末尾追加完整报告：训练 `9e80e541…`、推断 `2a95bad1…`。原 strict32 的 124 行、CID/PID、候选、答案、文献隔离和监督目标保持；native CPU 全文审计 `5d295ec5…` / 行审计 `80f5bbce…` 已真实完成，直接绑定，不重跑。

训练 `train_medgemma32_fullcontext.py` 复用原 `3daa85b0…`：显式新长度拒绝上限 14000，原 trainer 的 8192 未改；batch1、积累8、每 epoch 15 个8样本组加末4、16次更新、两 epoch 共32次、2 warmup、r16/alpha32/dropout.05、seed20261023、BF16/SDPA 和 native CE 全保持。加载 fresh 公开基座并新建 LoRA；只有本次同目录完整 epoch checkpoint 能恢复，不加载短 Med adapter。

原 `checkpoint=Path(args.resume)` 到整个训练终点/异常记录的完整代码逐字节相同，包含模型载入、adapter、AdamW、scheduler、Python/NumPy/Torch/CUDA RNG 恢复、epoch 数学、checkpoint 文件、fsync、暂停和 complete。10 个辅助函数 AST 相同。原 `training_complete.elapsed_seconds` 仍只表示该进程耗时；分 epoch 退出/恢复时，owner 以各 segment 和共享绝对预算统计总耗时。CPU 合同仅核 metadata 和状态文件 SHA，未声称 GPU 数值恢复等价。

原标签链 `preparation-manifest.json`（`356109e9…`）仍指向原训练 `cb7f67e1…`。新 fingerprint 的 `train_sha256` 指全文 `9e80…`，另在 `append_only_provenance` 同时保存原标签链与可逆追加来源。不能把旧 manifest 的训练 SHA 改写为新 SHA。新 schema `patientphex-medgemma15-fullbundle-text-relation-fullcontext-lora-v1` 加上源码、数据和来源指纹，阻止混用旧短 Med checkpoint。

训练入口使用既有全文 CPU `summary.json` 作为 `--cpu-reference`，不要求或伪造“新 trainer 已完整 CPU/meta 运行”。正式执行时实际 native 编码逐行核对既有 token/监督 SHA、完整消息 SHA、掩码与原生结束符，并明确记录 `fullcontext-cpu-reference-reuse.json`。124 条每 epoch 共 965436 tokens，监督总数仍 7150；所有输入不截断。

推断 `infer_medgemma80_fullcontext.py` 复用原 `87c187c8…`，严格绑定全文 80 条与原 80 条的可逆追加，原 candidates/row/PID 不变。新 schema `patientphex-medgemma15-fullcontext80-inference-v1`；固定输入 cap14000 / 最多新1600，原完整 model-load、generate、EOS 与响应保存段逐字节相同。实际编码逐行匹配既有全文 CPU 审计；输入总645000、最长13805，不截断、不选择子集。新 CLI 及保存源码清单见 `source-plan.json`。

`cpu-contract.json` 的20项是本地 source/input/resume 合同：真实124/80可逆重建、旧标签链替换拒绝、短输入混用拒绝、同32-call顺序与末4尾组、resume控制参数不改变语义配置、旧schema/坏Adam-RNG文件拒绝、同长度token与prompt变更反例等。所有正负检查已实际执行；未重核模型资产、未导入模型库。LR 表是原固定 API 的静态公式说明，不冒称执行了新 GPU scheduler。

评估由 owner 独立负责：固定 primary 全文 Med，唯一 secondary 为旧 SFT32 关系与全文 Med 关系并集，原实体/非关系不变；两 fullpred 同时冻结后各调用既有 v2 一次，强基线 `.660942…` 的既有指标复用，短 Med 不重评。不在本目录读取 gold、扫参数、改阈值、换组合或操作官方提交。
