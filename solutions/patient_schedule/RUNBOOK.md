# 恢复与执行入口

当前**四轮及四份持久ACK已完整，watch145075正常退出，epoch4 7c97bbdc已绑定F1 plan64df17bb**。不得重复部署、训练或watch；下一步仅原授权步骤6，随后实际终态步骤7。单一状态入口[evidence/ready.json](evidence/ready.json)。

外部B=`/workspace/patientphex-runs/round2-20261023/research/patient-schedule-fresh4-v1`，C=`$B/controller`；源码S为本目录。root已明确训练/完整F1授权，已按实际决定将 `GPU-AUTHORIZATION.template.json`另存为`GPU-AUTHORIZATION.json`，填写已给出的授权，原模板保持。实际authorization/plan已SHA绑定并推送。controller检查plan SHA及固定预算。若改卡/路径/预算，应另冻结独立plan与部署，不覆盖原冻结件。

1. `python "$S/controller/operations.py" deploy --directory "$C"`
2. `python "$S/controller/operations.py" preflight --directory "$C"`：实际全部15资产SHA、完整UUID空卡、SHM/.100空间；不因旧释放记录跳过。
3. `python "$S/controller/operations.py" start-epoch --directory "$C" --epoch 1`：唯一启动；先保存intent。出现UNKNOWN时查原intent/remote driver/job，绝不再start。
4. `python "$S/controller/watch_authorized_campaign.py" --directory "$C"`：只运行一个本地watcher，记录PID。逐轮inspect→完整导出→ACK→下一轮；四轮共享原deadline。观察失败停止watcher而不杀健康训练；按原已发生动作恢复，不能盲重开watcher。
5. 仅当 `campaign-complete.json`及四份ACK到齐，`python "$S/inference/bind.py" --controller "$C"`。它只绑定真实第四轮adapter，不加载模型；未齐会拒绝。
6. `python "$S/inference/operations.py" start --controller "$C"`，后续同入口`inspect`、终态后唯一`collect`。原80全fresh、1200+30。失败输出同样收件，无自动补生成。
7. 完整80成功收件后，`python "$S/quality/final.py" --plan "$S/quality-plan.json" --controller "$C"`，两预测先共冻，再各1v2；不可重复执行。旧baseline已在B/matched-baseline，禁止重跑baseline.py。

终点保留四epoch ACK、原campaign clock、resume-state-verified-001/002/003及全部raw/IDs/quality；独立增量终审。模型state在SHM/.100，Git只保存source/小指标/manifest。收件UNKNOWN只续同字节块，不重新NN或生成新原始响应。每阶段及时记录并push。

CPU准备若要复核：`cpu_contract.py --controller C --output 新路径`只读已固定本地输入/审查身份与小逻辑状态，不触GPU。现有结果已经完成，不为展示重复运行。`prepare.py`仅在新空目录打包，不网络；已备deployment直接用，不需再次创建。
