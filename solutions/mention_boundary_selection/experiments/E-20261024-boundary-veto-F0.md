# 同头 F0 BOUNDARY veto

目标：验证 root 独立 F1 veto 的小正增益是否在 F0 同样成立。F1主头 union/new-only 两负结论不变；本实验不重新训练，不重复这两个端点。

来源已核：3470 训练仅 F2/F3；578 NO 与539 BOUNDARY来自两份 train16 互换 OOF候选，联合排 F0/F1。旧F1 inference proposer48含 F0，仅属于旧推断候选；当前替换为强F0排0的1085原实体跨度。所有Top8来自公共41099银行，无别名字典或gold插池。

固定公式：1085条逐行同prompt/同candidate顺序/同 final epoch2 head/同greedy514状态语法。只有完整EOS、未cap且strict_parse为BOUNDARY时删除strongB同跨度的全部实体对象；其它状态保持原实体。关系、患者及其他对象不变。基线Score .7405721404401228，唯一预测完整冻结后最多1v2/旧0；零改变继承字节和指标，0v2。

373缺失Sap queries原生最大12 tokens，max32无截断；实际11.010秒完整，16:57:09.811953 UTC释放。165缓存＋373新query覆盖538唯一词面，全1085 native CPU6.673秒通过：616583 tokens/max698，模板与IDs原生HF逐项相同。CPU24小成员全部SHA回收，旧514语法合同复用，不重复训练检查。

17:03:03.946223763 UTC唯一推断开始，proc_9320b6db/PID2756581，.100 GPU1 UUID fc5d0525；源da63ebb2已push并核远端。独立1800+TERM30，过期入场与exclusiveclaim阻重复。按root就绪优先，经sequence确认无GPUstart后接卡，完成即交其GP链。

运行证据见 ../evidence/F0-running.json；外部目录 `/workspace/patientphex-runs/round2-20261023/research/mention-boundary-F0-veto-v1`。当前没有F0质量结果，未原dev/官网。不可将CPU合同或预测状态数称为得分。

## 完整终态

17:15:54.185579全部1085完成，770.097980秒，原wrapper770.969564秒/child rc0；17:15:54.981426释放GPU1=0MiB/noapps，已交sequence下一条GP链。991HPO/90NO/4BOUNDARY，0拒绝/cap。实际同final adapter504逐值exact。

所有2202推断小成员SHA完整回收，ZIP1184973B/160a1178；与20查询/24CPU原件形成完整证据。17:17:21先冻完整veto预测351f651c，再唯一v2/旧0：Score .7404920258945116，基线 .7405721404401228，delta −.000080114545611。删4=1TP+3FP；menF1 +.000487900766、docF1 −.000808358948，mic/mac逐值不变。原预测/阈值/所有4例同一规则，没有个别删选。

因此固定F0正门失败。F1小正不能代表稳定收益，本头以错误边界拒绝换取少量FP减少时仍可误删正确实体。绝不把3/4删除精度直接替代最终指标，也不因微小下降额外换cut或挑文献。源及同模型工程正常，不是超时/缺失输出造成的负结论。稳定terminal864c1e24不修改，后继backup/review只补pointer。
