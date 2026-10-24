# SCST 固定4×2探针：单一接手入口

**2026-10-24 17:21 状态更新（优先于以下准备期历史）**：原776b已在17:09唯一启动，69秒后因loading-info set JSON记录失败，0forward/0生成/0FB/0Adam，完整原21容器证据已收。下方776b启动命令现在仅为历史，严禁重启。根批准窄metadata修复后的独立fresh1800+30，旧17:39:49截止不变。最新入口为 `evidence/loading-recovery-binding.json`、`evidence/loading-recovery-deployed.json` 和外部 `bound-A100-loading-recovery-v2/binding.json`（ef629830，Sap已资源修订至同hostGPU1）；10件已部署全SHA，第一次仅readonly fresh查询UNKNOWN，尚无admission/start。实际5.20.2 tiny CPU调用UNKNOWN不重执行，root已明确原G serializer exact AST及本地类型合同足够闭合此已知bug，未知调用如实保留。下一统一恢复窗17:30:45，只读fresh成功后才发唯一新start；必要Sap等GP真实release，本地等待计入主钟。

2026-10-24：**源码、CPU合同、部署和增量源审已完成；没有admission、start、SCST GPU输出或正式训练。** 根线程要求暂时释放agent槽；收到 ontology_recover_0518 的seed39实际释放后唤醒本owner继续同一方案，无需另问逐步许可。

先读仓库 `AGENTS.md` / `HANDOFF.MD`，再读本文件。不要开启新的研究路线、下载权重或重复旧合同。

## 唯一执行身份

- Worktree：`/workspace/patientphex-worktrees/research-patient-set-risk`；branch `research/patient-set-risk`。
- 本线目录：`solutions/patient_set_risk`；本线外部产物根：`R=/workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1`。
- **唯一当前binding**：`R/bound-A100-v4/binding.json`，其中执行计划 `execution.json` 4420B，SHA **776b98e0459afc68b1600d0b543ca8c10eb8bca4e8b4ee5123790c1bbbbd08dd**。
- 当前 `pilot.py` 22711B，SHA **f7c9f813c031efa2b1e56fa5fba755180b3b2bd17e9160a8097c976b7df04979**。
- `.12`远端新namespace：`/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/patient-set-risk-v1/pilot-A100-v2`。
- 当前部署完成：`R/deployed-A100-v4/complete.json`，10件全fsync/reopen/fullSHA；短指针 `evidence/deployed-A100-generation-reset.json`。不要重部署或改文件。
- 原v3执行91b6c6ae及旧pilot-A100-v1 namespace保留原字节，**从未启动且现已被v4替代**。旧首inputs UNKNOWN已由原路径完整SHA恢复，不重写；当前v4完整成功。
- 固定4 inputs `rows.json.gz`：86878B/SHA **1c78f25a8a5c2d60a52b1687673d5ce76609f349af4e776e91961002cbda9862**，无held输入。

## 已授权目标及禁止扩展

原proposal `pilot-plan.json` f423d747保持原字节；`activation.json`记录root事前修订：至少2组实际非零advantage，且至少1组gold非空、两条输出均合法、概念集合F1不同；另一组可是真实格式有效性差异。

固定患者原索引 `[76,44,83,19]`；每患者greedy一次、fullcategorical sample一次；seed20261024+索引，T1/top_p1/top_k0/min_pNone，1600新token，无截断/修补/重采样。原G train32 d52父不变，8条生成与奖励全部完成前不更新。所有818 gold原子保留，原public HPO/Sap FP16+b59完整标准化；新quote不得漏查或用CPU FP32替代。

供给门过后：`mean_i[CEmean(original gold) - detached(A_i)/(11884/87)*SUM_logp(sample suffix)]`，真实EOS计入；8次串行FB、1次AdamW（lr1e-6、β.9/.999、eps1e-8、wd0、clip1），状态可丢弃。门不过则0FB/0Adam负关闭。统一**1800秒TERM+30秒KILL共享绝对时钟**，包含8条生成、等待/标准化、重放和完整状态持久化；必要Sap局部最多180+30且不得越过原共享钟。

没有87正式训练、held推断、v2质量或官方提交授权。当前best不变。不得减样本、调门或延时来过关。

## 资源条件与已有实际证据

主体只用 `.12 A100 GPU2`：UUID `GPU-2c22a952-40e2-952f-854c-ff668b5c9dc1`，mount `476f81cf-09e7-445b-a7f0-d0e12b12e50f`。准确前序owner是 **ontology_recover_0518**；其seed39于16:42:47已启动，必须等真实完整GPU释放。schema_extraction管的是另一台.100GPU1，不能用其release。

`.12` actual CPU库存已核：Python `/dataset/miniforge3/envs/vlm-det/bin/python`，PYTHONPATH `P/case-retrieval/qwen35-preparation/runtime`；Torch2.6.0+cu124、TF5.20.2、PEFT.21.2、tokenizers.23.1；native b4b17fe0，无FLA，CPU核查CUDA未初始化。原G d52父是 `P/grounded-generation-v1/formal-v1/output/epoch-002/adapter`。公共base在 `P/case-retrieval/qwen35-preparation/models/Qwen3.5-9B`。详见 `host-A100-generation-reset.json`。

Fresh gate要求目标无apps、used≤64MiB、实际free≥38000MiB、持久盘≥1GiB。原full87实测reserved31.39GB；新10423输入/1600suffix约37GB只是代理，不是已过显存证据。OOM保原响应停止，不缩短输入。

必要public Sap用 `.100 GPU0` UUID `GPU-126293b9-6677-8464-cee7-01e693efbf29`，root已优先留给本probe。sequence F047+27Sap整链于16:36:26.770621实际释放，原wrapper338ce0e1/child2754896；依据 `R/Sap100-prior-release.json`。真启动时query_driver仍fresh核卡，不能把旧release当持续空闲保证。不碰.100GPU1、.3或当前外部占用A800。

## 接续命令

以下均在本worktree运行。**只有收到seed39真实release后**，在 `R/A100-prior-release.json` 保存可核的真实wrapper/观察来源，至少含 `whole_chain_released:true`、上述准确`gpu_uuid`、时间和source pointer；不得提前编造release。

```bash
python solutions/patient_set_risk/remote.py start --binding /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/bound-A100-v4/binding.json --release /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/A100-prior-release.json --output /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/start-A100-v1
```

此命令fresh核卡，绑定新300秒admission，排他创建 `bound-A100-v4/unique-start-intent.json`，单次start。交接时该intent不存在。UNKNOWN必须按原intent/namespace恢复，绝不另start或重置时钟。

正常运行期间按合理间隔使用 `remote.py status --binding <同binding> --output <新观察目录>`。等待资源时不远端轮询；由owner明确交接。生成结束出现 `terminal/reward-request.json` 后：

```bash
python solutions/patient_set_risk/remote.py fetch8 --binding /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/bound-A100-v4/binding.json --output /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/raw8-v1
python solutions/patient_set_risk/reward_bridge.py --binding /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/bound-A100-v4/binding.json --raw-dir /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/raw8-v1 --sap-release /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/Sap100-prior-release.json --output /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/reward-v1
```

bridge核固定source manifest和8raw SHA，CPU准备，非零必要queries时绑定独立Sap namespace/局部admission/原共享deadline，单start并完整收件；0query不启动模型。最终reward先冻结，再先delivery、后reward原子发布给等待中的同一父模型。控制器UNKNOWN不自动重试；按原输出/完整SHA恢复，不能漏query或伪造reward。

真实driver/child均退出后：

```bash
python solutions/patient_set_risk/collect.py --binding /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/bound-A100-v4/binding.json --output /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-v1/collection-v1
```

collector要求实际noapps/driver和child absent，完整小包fsync/reopen/下载/memberSHA。若做过Adam，大adapter/optimizer/RNG完整状态留原持久服务器、重新全SHA，在receipt明确“没有下载大件”。最后将本线source/raw/reward/短报告的最小持久包闭合；临床raw、预测、权重不入Git。更新本线WORKLOG/REPORT/NOTES、小evidence、commit/push核远端；将实际结果和资源释放回报root，不改主台账。

## 已完成审查及实际解释边界

- 窄源审：`/workspace/patientphex-runs/round2-20261023/validation/patient-set-risk-scst-source-review/review.json` SHA483db3ba…，无剩余实质阻断；只源/既有tinyCPU。
- root配置增量审：同目录 `generation-config-delta-review.json` 2424B/SHA **b37002558506a45630937fba0d4a4bcee84a35356e05034aad733e31f05b48e2**。
- 新接口CPU证据：`R/generation-config-contract-v1/result.json`379B/SHA8b3a1fc4…；actual5.20.2 None字段会继承模型默认，故保存原默认后显式重置underlying默认，再核每条prepared配置和唯一无修改observer。不要盲传已无显式支持的use_model_defaults开关。
- tinyCPU loss原source通过，no-grad/grad maxdiff0；独立BF16串行/联合backward可有约.00097656舍入差，未冒称逐位等价。后续只修非有限记录，不重跑旧数学。
- 真实生成/回放统一BF16/cacheFalse、禁全部dropout；回放必须train+nonreentrant checkpoint并记录真实grad-enabled调用。same-full no-grad/grad沿原1e-5门。cached生成与完整回放保存全部逐token/sum差，不另杜撰1e-5门或用mask掩盖不一致；实际结果决定后续正式可用性。非有限先保存finite/count/null或明确字符串再停止。
- 以上都**不是9B工程PASS或分数提升**。最终真实供给可能为负，按预锁原样关闭；不为填GPU开新配方。
