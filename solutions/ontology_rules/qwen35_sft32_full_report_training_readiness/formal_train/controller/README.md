# 唯一两轮正式训练与持久状态控制

源runner86d1297c、training-plan0b97048c；controller-plan48ade338、entryb0d474cb。先deploy与fresh15baseSHA/UUID/SHM/target空间，再start-epoch1一次时冻结整体8400+TERM30绝对deadline。epoch1完整16step停止/释放→server1/fetch1各一次（共享min600/剩余全局时间）→.100完整SHA/fsync/重开ACK回传原SHM→start-epoch2恢复全部Adam/scheduler/RNG/globalstep16→step32/finalcheckpoint→同法完整ACK。

controller各动作先保存排他intent，source/target wrapper真实返回和deadline独立落盘。所有模型作业和传输未知状态先inspect原job/日志；不重复start，不重置预算。每个stage只能使用既定namespace和已录job。服务器只提供一个complete checkpoint白名单，owner在ACK后停止原recorded server；独立deadline同时约束server/fetch生命周期。

state全部在新私有/dev/shm/patientphex-qwen35-fulltext32-1045-20261024/formal2ep，.100持久目录ontology-round2/qwen35-fulltext32-formal-v1/checkpoints。项目盘仅源和少量metadata/log。恢复时先检查campaign、epoch-XXX-start-intent/job/state与export-intent/ready/fetchjob/ACK，不猜测是否运行。

新7项实际小文件HTTP+fsync/reopen CPU合同0.133461秒通过；本地UID为fixture模拟，真实远端UID/空间/大文件恢复不冒称已测。原69项runnerCPU合同另列；真实恢复digest仍由epoch2执行。该控制器没有推理或评分入口。
