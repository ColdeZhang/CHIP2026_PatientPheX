# 工作台账：患者集合风险优化

状态：CPU/公开资料准备完成，单4患者8序列供给与梯度probe提案已定，**未启动GPU或训练**。主分支基线21dca5f3，独立branch `research/patient-set-risk`、独立worktree；教师供给负结果留 `research/verified-rationale@d20e7634`。

目标：针对关系分项，判断完整患者概念集合F1的策略梯度是否提供旧CE/推断expected-F1/DPO之外的新监督。成功标准先是可区分、同义一致、实际采样可用的训练reward与正确梯度链，不拿训练成绩当验证成绩。

- 已读当前main AGENTS/HANDOFF/WORKLOG/MAP，核旧expected-F1仅推断cut，旧DPO固定同cardinality交换。
- 选择G87完整患者的SCST，只保一个方案；暂停BioModernBERT等另一架构调研。
- 原train32纯CPU：818完整v2 gold原子、5空患者，原target归一化mean F1 .9946025是监督供给自检；82删除/71外患者新增可改变reward，32不同quote同概念不改变reward。0模型/0fit/0heldgold/0quality。
- 初次Python/NumPy ABI导入失败在读数据前；固定系统Python3.13成功。最小四患者选择在任何GPU前按shape轴改成76/44/83/19，原记录保留。
- 来源及数学见 [REPORT](REPORT.md)，唯一参数/预算见 [pilot-plan](pilot-plan.json)。提出1800+30共享上限，非实际已启动deadline。
- 当前等待root审阅具体提案；即使准备通过也不启动87全量、174生成或held质量。A800 GPU2当前归sequence F0-47，必须实际交接后才可fresh门；新quote正常化需要原公共Sap完整查询。

实际CPU命令：

```bash
OPENBLAS_NUM_THREADS=1 /usr/bin/python3 solutions/patient_set_risk/census.py --grounded-source /workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation --run-root /workspace/patientphex-runs/round2-20261023 --hpo /workspace/patientphex-data/hp.obo --output /workspace/patientphex-runs/round2-20261023/research/patient-set-risk-readiness-v1/cpu-v3
```

现目录已存在，恢复读取现有结果，不为查重重算。全部训练案例125KB、公开页面/代码在仓外；Git只放源码、固定计划、短证据和记录。

## 2026-10-24 新 loss 与动态标准化源

根线程批准一次固定4患者×2轨迹、共享1800+30秒探针。原 `pilot-plan.json` f423d747 保留，新增 `activation.json` 记录事前供给门修订：至少2组非零优势，其中至少1组gold非空、两输出合法且概念F1不同；另一组可为真实格式有效性差异。没有采样后改门。

`pilot.py` 用fresh GenerationConfig和实际processor列表排除隐藏top-k/warper；只保存一份前步logits，记录全部实际token logp。生成/回放统一BF16 autocast(cache=False)，禁全部dropout；回放实际train+nonreentrant checkpoint并统计grad-enabled调用。完整8raw先落盘，当前父权重不变等待原FP16 public Sap，再按固定门决定是否执行8FB/1Adam；失败不重采样、不替换患者。same-full no-grad/grad沿原1e-5门，cached/full逐token差异单独记录，不宣称二者逐位等价。无formal87/held授权。

新增纯CPU tiny loss合同在.100既有Torch2.7.1运行，CUDA_VISIBLE_DEVICES为空、1CPU线程、50+5秒。真实同输入BF16 no-grad/grad maxdiff0；串行与联合BF16 backward最大差约0.00097656，均finite，明确独立舍入而非逐位相同。一次tiny Adam更新实际生效；这不证明9B CUDA成本或实际语义优势。指针 `evidence/loss-CPU.json`。

normalizer helper完成8raw→必要公共query→完整奖励接口及CPU合同（cb35c8f2），仅本子目录约93KB源和小证据；所有临床raw留仓库外。其GPUquery代码仅编译/静核，尚无新query模型调用。

资源：A800 GPU2已有外部PID46312，不抢占；主体待根线程提供真实空闲40GB/80GB窗口。.100GPU0先由sequence F047使用，Sap也需等待其实际release。此刻无SCST GPU启动、无新held gold、无quality。

## 2026-10-24 A100 主体绑定，无GPU启动

根线程确认A800三卡现为外部进程占用，主体改排 `.12 GPU2` 原seed完整链之后。纯CPU只读实际库存确认 Torch2.6.0+cu124/TF5.20.2/PEFT0.21.2/tokenizers0.23.1、原native b4b17fe0、无FLA、CUDA未初始化；原d52 adapter及公共Qwen3.5本地路径存在，项目盘约48.4GB空余。此项未重复13步或做新分词/模型调用。

`host-A100.json` 采用原同机G full87实际峰 allocated28.18GB/reserved31.39GB作依据，新10423完整输入和1600suffix只是约37GB代理，未冒称旧成本覆盖新形状。需实际>=38000MiB free且无process，真实probe才可确认显存。

新 `remote.py` 与 `reward_bridge.py` 绑定完整normalizer源码和主执行SHA，单次排他start/独立300秒admission，8raw→必要Sap→完整reward在同一1800秒原绝对钟内；query UNKNOWN不重启。delivery先冻结再原子发布奖励，主模型核原8raw和normalizer manifest。`bound-A100-v1`未执行；v2仅补完reward控制器source绑定，数据和数学不变。入口 `evidence/binding-A100.json`。
