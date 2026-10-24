# 当前运行与恢复

先读 main 的 AGENTS/HANDOFF/WORKLOG 和本路线台账，再核真实文件。这里不替代实际收据；相同任务不得重启。

`R=/workspace/patientphex-runs/round2-20261023/research/patient-balanced-mixed-loss-alpha05`，源码目录为本文件所在目录。A100 mount 为 `476f81cf-09e7-445b-a7f0-d0e12b12e50f`，GPU2 UUID 为 `GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1`。只有本任务 owner 查询此 mount；明确 LUNA 后按外部 `transport/offline-backoff-*.json` 至少 185 秒退避。

## 正式训练

唯一 `proc_afaf0b83-b325-446e-9769-5955d8c8136d`，driver 19546 / child 19564，实际 08:52:35.025475 UTC 开始。绝对 TERM 为 09:32:35.025472、KILL 为 09:33:05.025472；不因工具离线延时或重开。固定 train32 的 124×2、32 次调用、warmup2、真实尾组4、α=.5 和 hook7b99。探针状态已丢弃。

恢复只运行 `formal_remote.py observe --directory "$R/formal-v1"`，不要运行 `start`。所有阶段 intent、原字节部署包和实际控制器记录均在该目录；UNKNOWN 只表示工具结果不确定，先观察原目录/句柄。

原进程终态后，用 `formal_collect.py --directory "$R/formal-v1"` 回收。同一个 `.collection/training-metadata.zip` 只冻结一次，之后按 SHA 续读。所有非权重输出收回；两个 epoch 的 adapter 与完整 Adam/scheduler/RNG 留在原服务器持久路径，collector 实际 fsync/重开 SHA 并核对 ACK，记录 locator。不能把只有目录或只有 `training_complete.json` 当作通过；必须 `formal_reported_success=true`、两份 ACK、实际权重/状态 SHA 与 wrapper 全部一致。

原始 probe 失败72秒和独立修复成功159.844秒分别留在 `probe-v1`、`probe-interface-recovery-v2`，都禁止重跑。工程统一审7a010已通过；这不代表质量达标。

## 已授权的后继

最终 epoch2 完整持久确认后，只运行一次新模型 final80，预算600+TERM30。`final80/final_infer.py` 目前三 actual receipt SHA 为明确 UNBOUND；依真实 `training_complete.json`、`epoch-002/checkpoint.json`、`durability-epoch-002.json` 绑定常量，并更新 `final80/input-sources.json` 的相应 refs/source SHA。只做实际身份绑定，不改生成、strict parser 或原80输入。

新 `final80-plan.json` 应绑定9件 payload、原服务器三件实际训练收据、当前 driver/operations/source SHA，输出是新 `final80-v1/prediction`。`infer_remote.py` 的 `payload_sources` / `expected_remote_files` 可构建文件映射；权重留在原正式 epoch2，不能复制旧模型输出。源/计划先 commit/push，再 pack/deploy/fresh GPU gate/唯一 start。使用同一现有 Qwen 基座、最终新 LoRA、原贪心/非思考/BF16-SDPA/14000与1600限制，严格完整 EOS/PID/CID；生成 IDs 和原文先持久，失败保留原件且停止，不引入 partial/abstain 修复。

结束后 `infer_collect.py` 收原整个输出树；两次模型执行都不能因收件问题重跑。新 `quality/plan-template.json` 必须绑定真实完整 responses、manifest、wrapper 和已绑定 inference source，以及本地/远端 prediction 根；模板自身保持阻断。

`quality/assemble.py --plan ...` 先同时冻结完整16篇的 N∪R∪newP 和 newP-alone；前者替换旧P，不能把旧P独有关系保留成四方。1189实体、39患者以及 phenotype 外原字节保持，原587候选/1557对按原输入求和。只有共同 freeze 后才运行 `quality/score.py --plan ...`，30+TERM1、每端点唯一 v2、旧指标零重评。总Score门针对原三方 `.663841820660`；同时披露强旧全文 `.676546354663`，不得将低于它的结果称全局最佳。macro 无额外否决门；不自动迁 F0/原dev，不上传官方。

原dev best仍 `.754361398199`，不是本控制的F1指标。完成、失败和负结果都要更新路线台账并 push；模型/原始日志/全量预测留仓库外。
