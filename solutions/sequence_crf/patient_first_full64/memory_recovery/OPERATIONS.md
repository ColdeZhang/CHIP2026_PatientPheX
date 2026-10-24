# 新 selected-CE full64 唯一运行入口

原full64失败job无epoch，不可重启。新selected-CE训练proc7f51已06:03:05Z rc0/2991.911秒完成2ep/68calls，两fullstate持久。后继原dev64唯一推断proc05ce已06:13:10Z rc0/311.034秒完成，卡已释放；203件完整回收、freeze2ac4b470之后唯一v2实际完成，Score .746261042824（相对a179+.000179077837，3TP3FP、macro下降），未达.8且低root最新已独审全局3049d9fe .754216969282。**整条链已结束，以下命令只作历史恢复说明，不再执行start/bind/assemble/evaluate。** 新会话从WORKLOG与final-terminal-summary读取当前终态。

外部F=`/workspace/patientphex-runs/round2-20261023/research/patient-first-full64-memory-recovery/formal`；源S=`solutions/sequence_crf/patient_first_full64/memory_recovery/formal`。

原训练/推断watch均已终止；F/infer-0608-watcher-terminal.json、training-collection.json和inference-collection.json可只读验证。所有原小件/完整pred都已本地外存，完整状态与raw还在远端持久目录。终态审查复用原证据，不重跑模型、parser合同或质量。

1. 原训练job历史只读：`python "$F/monitor_existing.py" <新收据名.json>`。UNKNOWN≥60s、LUNA≥180s退避后一次已有状态读取；不由连接失败推断训练终态。权威远端timeout不依赖本地监控。
2. 完整wrapper后，优先用memory_recovery/collect_terminal.py --phase train --directory F，按固定小清单压缩回收并写兼容training-collection；包含新selected-loss-binding，不读取大state字节。必须2ep68calls、两durability/完整statefsync与SHA通过；首次OOM/非有限/截止保留并停，不改变量/重启。
3. wrapper0且两epoch持久通过，S/finish_setup.py bind-inference --directory F。从真实小收据填三常量，原infer.py070e模板保留，actual-final-infer.py另存Git final_infer.py，不覆盖冻结模板。提交push核远端；deploy-inference只两文件，UNKNOWN先库存缺件。
4. fresh资源核后S/formal_operations.py start --phase infer --directory F --checkpoint实际gitrevparse；900+30、原64全fresh、greedy/raw先存/有限子集codec，不重试。随后finish_setup bind-endpoint，完整203成员同collect_terminal.py --phase infer回收后endpoint assemble→共同freeze→evaluate唯一v2。
5. 唯一新质量仍a179完整pred literal∪患者关系，旧所有实体/关联/public-new保留。不得改用新best0e2c作本控制baseline，不评patient-alone，不读未冻gold或partial评分。Score严格升记候选、所有四项和macro变化披露，>=.8且终审通过才用户验收；无官方提交。

文件源码、训练/原dev输入/固定解码和质量合同已审来源可复用，不重跑整套旧CPU。新source差异仅选监督CE与失败row/step观察；probe权重不续用。每里程碑更新台账、commit/push，完整权重与raw/pred外存。
