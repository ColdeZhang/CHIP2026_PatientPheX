# Full64 条件正式入口与恢复

当前为源模板，**没有启动任何远端任务，也没有生成实际 GPU-AUTHORIZATION**。唯一主入口 `entry.py` 显式执行指定阶段；每个阶段仍使用原 exclusive intent/job/receipt。不存在“失败就重新训练”或自动重置5400秒窗口。

默认仓外controller：`/workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-formal-v1/controller`。

```bash
python solutions/patient_response_code/full64/controller/entry.py prepare \
  --controller <new-controller-directory> \
  --probe-terminal <actual-probe-wrapper-complete.json> \
  --cost-decision <actual-full64-formal-cost-decision.json>
python solutions/patient_response_code/full64/controller/entry.py deploy --controller <controller>
python solutions/patient_response_code/full64/controller/entry.py preflight --controller <controller>
python solutions/patient_response_code/full64/controller/entry.py start --controller <controller> --epoch 1
python solutions/patient_response_code/full64/controller/entry.py watch --controller <controller>
```

`prepare` 在创建目录前核新probe：240+30，12F/B/1Adam，207(4800/128)与137(6462/125)均过，finite/base冻结、真实释放、probe状态弃掉；cost必须准确绑定 wrapper/step/complete/原probe plan，complete→run-manifest→实际source/plan链也复核。估计须有限且0<estimate≤5400；估计不是保证。输入真实ref缺失或失败不会创建授权/部署包。需绑定的cost字段见 `activation.py`，不接受旧C32极值或旧预算。

源与.100持久根均为各自机器的 `project/patient-response-code/full64-v1`；训练输出只在 `/dev/shm/patientphex-response-code-20261024/full64-formal2ep`。两ACK global34/68；server/fsync/reopen协议复用。HTTP读取仅允许冻结checkpoint和少量_run文件；新增服务自身截止循环，外层仍固定 min600/剩余整体窗口。停止driver回执不单独证明HTTP子进程消失，实际进程释放仍由owner按PID/原截止核验。

初始watch已使用原 `watch_state_recovery` 的只读状态退避：LUNA_OFFLINE等待185秒，UNKNOWN首次只读30秒后185秒；每段等待不超过30秒。只有纯读取可重试。任何start/publish/transfer未知结果保留原intent、停止自动推进；先用 `entry.py inspect` 或原专用read-only恢复脚本确认，不重新调用模型。`source_bridge.py`、`bootstrap.py`、精确piece恢复、ACK原字节读回/补登记源码保留在新namespace，均不改变配方或截止。

可选显式阶段：`inspect`、`start-server`、`start-fetch`、`collect-export`、`collect-training`；第二轮由watch在完整第一轮ACK后启动。原本地watch死亡后用 `recover-watch --recovery <unique-name> --not-before <unix>` 接续原作业；不并发两个watch。备用bootstrap三阶段 `bootstrap-target/prepare/start` 只在部署/未知状态按原证据确认后使用，不与正常start重复。

两ACK与真实campaign-complete齐后：

```bash
python solutions/patient_response_code/full64/controller/entry.py bind-inference \
  --controller <complete-controller> --inference-directory <new-external-inference-directory>
```

随后使用相邻 `inference/README.md` 的纯推断操作。此处无gold、v2或官方提交入口。源准备由helper完成，本任务所有Git/远端/真实probe/训练/推断均交主线程。
