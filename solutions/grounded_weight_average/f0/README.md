# 固定 F0 复验运行入口

本控制沿 F1 同一 ed184ecc/r32/alpha32 合成适配器、原生 BF16 greedy/max1600、完整47全文和同 CPU FP32 Sap 查询策略。新47/16doc/947实体，唯一端点 `B4 ∪ (G_mean ∩ G_original)`；旧 B4 .7281956289113009 的343关系全部保护。只做一次新v2，byteexact则0；无单臂/原dev/阈值搜索。

`S`为本目录；`D=/workspace/patientphex-runs/round2-20261023/research/grounded-weight-average-v1/F0`；`G=/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation`。

当前仅原 `infer-v1` 在跑。完整native47 CPU收件 `native-cpu-v1/collection.json` 已过；本地只读watcher持 `infer-v1/watch/owner.lock`，只观察原作业。先看最新 `observation-*.json` 和 `watch/terminal-observed.json`；不重复start。进程/截止见 `../evidence/F0-inference-start.json`。结束后依下列依赖顺序接续，现有目录/文件意味着该阶段已执行，先恢复，不重跑。

1. `python S/direct_collect.py`：只读原complete/responses，按原SHA恢复94单行raw/IDs。没有remoteZIP完整性宣称，不重生成。
2. `python S/postprocess.py extract-bind`；在原G worktree PYTHONPATH执行 `python -m solutions.grounded_generation.f0.extract_quotes --plan S/extraction-plan.json`；再执行原 `G/f0/held_prepare.py --extraction-manifest D/quotes-v1/manifest.json --output-dir D/normalization-preparation`。
3. `python S/normalization_cpu/bind.py --preparation-manifest D/normalization-preparation/manifest.json --inference-collection D/infer-v1/collection.json`。若没有必要查询直接进入4；否则先保存/推送实际query plan，沿该目录 `remote.py pack --plan ... --directory D/Sap-CPU100-v1`、`deploy`、唯一`start`、`observe`、`collect.py`，使用 `--client-dir /workspace/CHIP2026_PatientPheX/scripts`。仅CPU FP32/16线程300+30，CUDA隐藏，不占用他人GPU。
4. `python S/postprocess.py normalize-bind`；原G worktree PYTHONPATH执行 `python -m solutions.grounded_generation.f0.held_finalize --plan S/normalization-finalize-plan.json`。
5. `python S/postprocess.py assemble`：所有完整source共冻；记录并推送freeze后，`python S/postprocess.py score` 仅一次，gold只在这里解锁。`mean-new-only.jsonl`是未单独评分的审计产物。
6. `python S/persistence.py build` 冻结增量entry/包；`transfer` 沿已实证F1同小I/O程序实际写入/fsync/回读逐SHA；完整正负结果均记录，交root/validation，不自动迁originaldev。

所有 source/plan/输出不可覆盖；工具UNKNOWN按同意图只读恢复。旧F1两次目录collector UNKNOWN与成功direct收件是旧已结事实，不能记成新F0实际故障。双父实际train32名单各与全部16篇F0交集0，见 `../evidence/F0-ancestry.json`；严格normalizer运行时也检查train32与输入文献不相交。
