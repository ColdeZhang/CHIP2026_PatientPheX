# 患者四轮统一时程台账

当前：**CPU/source完整准备完成，root现已分配A800 GPU2并授权完整4epoch+F1；源码先推，随后fresh核与唯一启动。尚未实际GPU。**。分支`explore/patient-schedule`，base main`4af0123`。只修改本路线。单一恢复入口：[evidence/ready.json](evidence/ready.json)，命令见[RUNBOOK](RUNBOOK.md)。

## 目标与固定方案

原F2+F3 patient-first32：124行/87患者，保持原标准native CE/输入/目标/候选、9B/r16/seed/AdamW1e-4，只改fresh4epoch/64call/warmup4。四轮完整SHM状态逐轮导出.100并全SHA/fsync/重开ACK，原统一时钟不重置。最终epoch4唯一完整F1，两完整预测共冻后各1v2。

主臂强全文N∪P4同时对比强N .676546354663及匹配N∪旧P2 .671703122962；诊断P4-alone也可凭超强N进入候选。按超过强基线的最高Score选、平分主臂，macro披露。不得用旧weak triple小升冒称globalbest，不扩F0/dev/官网。见[固定实验](experiments/PS4-20261024-v1.md)。

## 本轮已完成

- 事前源码/plan先推`6942dc2`；唯一授权匹配baseline CPU0.282s/rc0，pred `64bd69d8…`、metrics`97d5f3a3…`，Score .671703122962，比强N−.004843232。先freeze后1次v2，旧臂0；不重模型。
- trainer/controller/完整F1推断/固定双质量均有可执行源码和完整prepared22件payload。10核心数据/批次函数与原训练AST相同；原infer loader/generate AST、严格parser SHA相同；改动仅4/64时程、完整持久恢复及raw-first保护。
- CPU增量0.090s：10项+37负向/变异检查，4个tail4/64call/3恢复边界/全部原数据身份通过；额外UTF8非关系字节、compound字面集合、缺患者拒绝和4项候选选择合同通过。无新native分词/模型/GPU，真实旧审查复用，不夸成新GPU恢复证明。
- 建议训练含4导出5400+30、F1独立1200+30、CPU双评120s。依据同patient A800旧2ep1141.720s和既有59.36s/520MB导出代理；新4ep未实测。旧reserved80.753GB须整80GB卡，4state约2.1GB入SHM/.100，项目盘不写state。
- 大输入/完整预测/deployment均在外部`R/research/patient-schedule-fresh4-v1/`；Git只小证据/源码。初始未运行准备历史另存，不覆盖既有实际结果。

## 下一步及停止边界

root已收到新生成A100显存PASS并将A800 GPU2分配本路线；授权已另存evidence/GPU-AUTHORIZATION.json，原false模板保留；还没有start-intent或watcher。若获相同预算与资源授权，沿RUNBOOK一次部署/fresh核/唯一epoch1，后续4ACK再绑定最终80。工程失败保原raw/状态，不盲重启、不自动扫epoch。完整质量再做一次实际增量终审；本轮无新patient4质量结论。

Git：上一个已核远端检查点`6942dc2`；最终prepared检查点将在本页完成后同批commit/push，最终hash以Git与交接消息为准，不为记录自身hash追加循环提交。
