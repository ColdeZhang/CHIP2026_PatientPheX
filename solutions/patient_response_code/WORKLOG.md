# 工作台账

当前：新分支explore/patient-response-code，base main cd4086b；正在冻结并推送新H51，尚未启动GPU。只改本路线。

- 目标：新响应code表示能否超过同任务强N .676546354663。主coded-alone、诊断N∪coded，固定两全量先共冻后各1v2/旧0，最高端点超过N可候选；另报历史fullsupport .686338981，非原dev。
- 数据：短train124 cb7f→f73cb，全文held80 2a95→54c8，原user整段UTF8/CID-HPO/clinicalPID/mentions不变，旧renderer0调用。原样本顺序保持；只是新的system输出说明＋response表及target患者值。
- 实际CPU：18.607秒PASS，训练427560tokens/max5910，监督7170→8138（PT001四token）；held685372/max14570，新工程cap14848不裁切，gen1600。所有target逆映射精确，12负例拒绝；旧identity桥接非本机制。来源R/research/patient-response-code-readiness-20261024/ready.json。
- 探针：A100GPU2 UUID2c22a952；12行first8建立非零诊断Adam，然后3条累积（含max监督128 index94）＋最长5910 index6；诊断不复演formal warmup，丢弃状态。240+TERM30；复用实证indexed标准CE/loader，A100Torch2.6/runtime明确区别CPU A8002.7。
- 正式条件：fresh原2epochs32calls/warmup2/标准CE/r16/seed，含两完整SHM→.100 ACK统一3600+30；原逻辑state恢复不重置时钟，probe状态不用。最终全部80fresh，只有最终epoch2质量，不选epoch。
- 外部R/research/patient-response-code-v1/。原始数据/模型/完整输出外存，main与旧patient分支未改。source/plan先push后唯一start；UNKNOWN查原作业，不盲重启。
