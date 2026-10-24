# 新 selected-CE full64 唯一运行入口

原full64失败job无epoch，不可重启；本次是新fresh。root已统一授权从完整训练推进到既定唯一dev终点，不需逐步批准。当前job `proc_7f51657c-95c7-4118-9d9d-68377242d294` / driver141616，A100 GPU2 UUID2c22，05:13:13Z，4500+30；近似TERM06:28:13/KILL06:28:43，实际driver-started为准。已有launch-intent，不得重复start或扩大预算。

外部F=`/workspace/patientphex-runs/round2-20261023/research/patient-first-full64-memory-recovery/formal`；源S=`solutions/sequence_crf/patient_first_full64/memory_recovery/formal`。

1. 原job只读：`python "$F/monitor_existing.py" <新收据名.json>`。UNKNOWN≥60s、LUNA≥180s退避后一次已有状态读取；不由连接失败推断训练终态。权威远端timeout不依赖本地监控。
2. 完整wrapper后，用S/formal_operations.py collect --phase train --directory F；若大响应失败可按同一固定小文件清单zip压缩回收，不重hash大state。必须2ep68calls、两durability/完整statefsync与SHA通过；首次OOM/非有限/截止保留并停，不改变量/重启。
3. wrapper0且两epoch持久通过，S/finish_setup.py bind-inference --directory F。从真实小收据填三常量，原infer.py070e模板保留，actual-final-infer.py另存Git final_infer.py，不覆盖冻结模板。提交push核远端；deploy-inference只两文件，UNKNOWN先库存缺件。
4. fresh资源核后S/formal_operations.py start --phase infer --directory F --checkpoint实际gitrevparse；900+30、原64全fresh、greedy/raw先存/有限子集codec，不重试。随后finish_setup bind-endpoint，完整203成员回收后endpoint assemble→共同freeze→evaluate唯一v2。
5. 唯一新质量仍a179完整pred literal∪患者关系，旧所有实体/关联/public-new保留。不得改用新best0e2c作本控制baseline，不评patient-alone，不读未冻gold或partial评分。Score严格升记候选、所有四项和macro变化披露，>=.8且终审通过才用户验收；无官方提交。

文件源码、训练/原dev输入/固定解码和质量合同已审来源可复用，不重跑整套旧CPU。新source差异仅选监督CE与失败row/step观察；probe权重不续用。每里程碑更新台账、commit/push，完整权重与raw/pred外存。
