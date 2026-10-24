# Full64 原正式作业历史入口（已 OOM 关闭）

**原作业已于04:51:51 OOM/rc1，无完整epoch；下列成功后继命令仅为冻结历史协议，当前不得执行或重启。新显存恢复在独立目录，工程探针与完整重训授权分列。**权威当前状态看路线 WORKLOG；这里是操作顺序，不能把历史 job 启动回执当仍存活。

所有命令从本 worktree 执行，先设任务变量：

```bash
F=/workspace/patientphex-runs/round2-20261023/research/patient-first-full64-formal-preparation
S=solutions/sequence_crf/patient_first_full64/formal
```

A100 mount476f、GPU2 UUID2c22；不要误切其他卡。当前训练 job `proc_5b00069a-f26c-45c5-a559-c9aa3e7d1d64`，04:42:33Z，4500+30；近似TERM05:57:33/KILL05:58:03。**已有train-launch-intent，不可再次start。**

1. 正常只读：`python "$F/monitor_existing.py" <唯一新收据名.json>`。LUNA_OFFLINE后至少180秒退避；UNKNOWN先核已有小状态，不重写/重启。实际停止边界在远端driver，与本地连接无关。
2. 完整wrapper出现后：`python "$S/formal_operations.py" collect --phase train --directory "$F"`。不完整仅报告已知状态，不评分；非零退出/失败保留并报root。成功必须2ep/68calls、两个完整状态fsync/祖先链/回读SHA通过。
3. 成功后：`python "$S/finish_setup.py" bind-inference --directory "$F"`。只从真实小收据填3SHA，产生`actual-final-infer.py`及`infer-plan.json`。**保留Git中的infer.py20b7模板不变**，把实际新源码另存Git `final_infer.py`，防旧引用漂移。不得再次绑定/覆盖已有输出。
4. 将实际新source/plan/receipt提交推送核远端。`python "$S/finish_setup.py" deploy-inference --directory "$F"`只部署远端infer.py+infer-plan.json两小件；逐文件原子，非跨文件事务。UNKNOWN先inventory这两文件及`.final-bound.tmp`，仅续缺件，不能重复mutating调用。
5. 部署实际SHA通过后：`python "$S/formal_operations.py" start --phase infer --directory "$F" --checkpoint <真实已推40位SHA>`，fresh gate、900+30、原64全fresh，无缓存/重试。随后`python "$S/finish_setup.py" bind-endpoint --directory "$F"`绑定实际infer-job及3训练收据；保存endpoint-plan并推送。模型参数/解码/codec不能改。
6. 原infer wrapper完成后用`python "$S/formal_operations.py" collect --phase infer --directory "$F"`，成功应有203小成员（4control+7output+64×3），完整64。硬失败保留全部raw，停止并报root；不得评分partial或续生成。
7. 完整成功后：`python "$S/endpoint.py" assemble --directory "$F"`（30秒）→完整raw/单一pred共同freeze→`python "$S/endpoint.py" evaluate --directory "$F"`（60秒）。最多一次新all/text v2，只a179 literal∪patient；不评patient-alone，不扫阈值/组合，旧metric复用。不得覆盖既有assembled/evaluation目录或原失败证据。
8. 完整结果/真实成本/产物SHA提交推送，交validator一次终审。Score严格升为candidate best，所有分项和macro下降如实；Score≥.8且独审格式/来源/复现通过才达到用户合并门槛，无官方提交。

`infer.py`与endpoint动态收据未齐会主动阻断。模型/完整state/全raw/pred留外部；Git只保存小源、计划、关键终点和清单。反复读取大权重SHA或重跑既有CPU/parser合同无必要；真实加载入口仍按原规则核资产身份。
