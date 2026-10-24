# 显式阶段接续入口

`stages.py` 是显式阶段 CLI。父代理接手后已补两个实际合成故障合同：启动RPC响应丢失后第二次调用不再发RPC；损坏ZIP在解包前拒绝。合同没有调用MCP/GPU或质量。既有downstream的24项模型/装配合同单独保留。

目标：接续唯一 core 作业，实际收齐完整训练 ACK 后，运行固定 80 条推断和唯一替 Q 四源质量终点。模型和质量逻辑完全复用 `../downstream/`，不在此重写。

所有命令均由父代理显式按序执行，不运行后台 watcher、不自动跨阶段。入口默认 `core_status`，只读原 core run。例：

```bash
python solutions/patient_swap_dpo/monitor/stages.py --directory /workspace/patientphex-runs/round2-20261023/research/patient-swap-dpo-v1/deployment-v2 core_status
```

后续将末尾模式依次换为：

1. `collect_core`：读取 driver、formal complete/ACK/assets/checkpoint 原始小件，核三阶段正常退出、释放 GPU、实际 execution-plan、完整 10-step/checkpoint 身份。
2. `bind_infer`：调用既有 `downstream/bind.py infer`，实际完成字段再由 binder 核验。输出 `D/downstream-v1/inference-plan.json`。
3. `deploy_infer`：部署被冻结的 7 个源码、held80/prompts80 两个输入和绑定 plan，每件最多 512 KiB；每次先保存 intent，再上传并逐字节回读。已有未知上传只能读原路径恢复，不能重写。
4. `start_infer`：仅一次启动独立 `downstream/driver.py`；1200+30 秒仍由既有 driver 执行。持久化启动 intent 后才调用 `start_process`；出现未知 RPC 后再次运行会拒绝启动。
5. `infer_status`：只读原 `run/infer80`。可重复手动查看，不启动任何模型。
6. `collect_infer`：按 actual terminal-receipt 分块读取完整 ZIP（每块最多 512 KiB），核 ZIP SHA、全部 inventory 成员与路径、实际绑定 plan、formal complete/ACK，再解包。读取中断保留 partial，下次从已有偏移续读，最终整包 SHA 必须一致。
7. `bind_quality`：完整 raw/native 收齐后才调用既有 quality binder。
8. `assemble`：唯一完整预测冻结，不读取 gold。
9. `evaluate`：明确调用唯一评分阶段；要求 freeze 存在。既有质量脚本负责零语义差继承 exact B4/0v2，否则固定 1v2/old0。此模式会读取冻结 F1 gold，仅由父代理在授权阶段执行。

所有新小件、意图、回执及大 ZIP 都放仓外 `D/downstream-v1/`。任何本地绑定/装配/评分阶段有 intent 就拒绝重复执行，应检查原输出恢复；不删除 intent 以绕过防重复机制。已收集文件只能复用相同字节。原 core 训练、reference/probe 不会由此脚本启动或重启。

部署额外核全件SHA、fsync文件及至本任务根的父目录链。源码准备与合成故障合同不代表实际推断或质量通过；真正调用后以intent/receipt记录为准。

`watch_core.py` 是另一独立的只读观察器，只查询已经启动的core，不启动模型或跨阶段。必须以独立session启动；ready记录PID/PPID/SID/start_ticks并持exclusive lock。正常60秒观察，首次读失败后30秒、后续185秒；未知状态不记为模型失败。
