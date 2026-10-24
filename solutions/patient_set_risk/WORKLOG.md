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

窄源审发现未触发的非有限诊断缺口：原JSON allow_nan=False可能在保留NaN/Inf norm前异常。GPU前机械修复为finite/count/null（逐token非有限值用明确字符串），保存首非有限logits shape/count和实际IDs，再按原finite/1e-5门失败；没有改损失/阈值/患者。原CPU数学 source身份仍保留，修复仅记录路径，未重跑旧数学。最终未执行绑定v3替代v2，旧文件都留仓库外；source审只核此diff。

A100小文件部署完成：首rows.gz write_file返回UNKNOWN，原路径只读完整SHA确认实际已写86,878B/1c78f25a，不重写原未知路径，只补其它9件；随后10件全体fsync/reopen/fullSHA闭合。未生成admission或start。`evidence/deployed-A100.json`指向完整恢复链和源独审483db3ba。

sequence的F047+必要27Sap链已于16:36:26.770621释放.100GPU0，原wrapper338ce0e1/child2754896，保真实release pointer；Sap真正启动仍须fresh gate。A100主体继续等前序seed完整训练/推断链，不能把其它卡释放当作本卡可用。

## 2026-10-24 实际生成配置默认继承修补

等待资源时只读 actual TF5.20.2 `generation/utils.py` 7944240f：`_prepare_generation_config` 会把None字段从 `model.generation_config` 合入，即使传fresh config；这个版本不提供可盲传的 `use_model_defaults=False` 显式参数。原processor门会拒隐藏warper，但实现“不继承”还应显式清空模型侧默认。

无模型CPU针对性实际API合同复现：synthetic祖先min_p=.3在caller min_p=None时被继承；把underlying默认重置为仅原EOS/pad/bos的fresh config后，min_p=None，固定T1/topk0/top_p1原值不变。CUDA未初始化，未实例化/下载模型。`pilot.py`现保存原model config与每条actual prepared config，核有效参数和唯一不改logits的observer；根线程认可机械修补，不改采样配方、预算或门。小证据 `evidence/generation-default-contract.json`。

原91b6执行绑定与已部署v1 namespace完全保留、未admit/start。新增v4绑定使用pilot-A100-v2新namespace，原fixed4 inputs相同。没有重跑旧loss/normalizer数学或另做整套源审。

v4新namespace的10件部署完整fsync/reopen通过，原v3部署完全保留。新inputs仍为86,878B/1c78f25a，无输入变化。`evidence/deployed-A100-generation-reset.json`是当前部署入口；没有admission/start。准确前序owner为ontology_recover_0518，须等待其seed39真实释放.12GPU2，schema_extraction只管另一台.100GPU1，不能混用释放信息。

根线程确认seed39已16:42:47启动、尚需约1105秒，要求释放agent槽等待owner事件，不开远端watch。最终单一接手入口 `EXECUTION_HANDOFF.md` 已写，含776b最新完整部署/源码/root增量审、真实资源条件、单次启动→8raw→动态奖励→收件准确命令、UNKNOWN恢复边界。此刻local unique-start-intent仍不存在；没有admission或GPU开始。收到ontology实际release后由root唤醒本owner沿同意图继续。

## 2026-10-24 17:09 实际唯一探针启动

root唤醒后沿776b/f7c9同计划恢复。先核ontology seed39实际17:04:46.618734释放.12GPU2（ec7da058），再fresh资源门；唯一proc_cc308f8f、driver85653/child85669于17:09:49.159650启动。原硬TERM **17:39:49.159593 UTC**，KILL **17:40:19.159593 UTC**，不重置。

当前没有formal/held/quality。首观察尚在CPU/加载；精确源/意图/时钟见 `evidence/actual-start.json`。必要Sap前序更新为ontology206实际17:09:44.596978释放的fee6ee8f，使用外部 `Sap100-seed-prior-release.json`；原sequence receipt保留，不覆盖。后续reward_bridge应传这个最新release路径。

## 2026-10-24 17:10 首次工程失败及窄恢复

原driver17:10:58.188 rc1/69.028669秒，actual peak17.91GB基座已装载；`pilot.py:66 -> common.py:18` 在保存loading-info中的set时JSON TypeError，尚未载父LoRA、0forward/0生成/0FB/0Adam。原G raw_store有集合JSON支持，我迁移小写入器时漏了该策略。完整21容器/20payload已roundtrip全SHA，原partial loading.json.tmp及trace保留。

只复用原G `json_default`（set/frozenset→sorted(key=repr)，未知类型仍TypeError），local AST逐节点等同原G；allow_nan=False和原missing/mismatch门不变。本地类型/拒绝合同通过。actual5.20.2 tiny随机CPU checkpoint类型合同0forward/无预训练下载，第一次源码write UNKNOWN已完整SHA只读恢复；后续50+5 CPU命令和一次只读目录均UNKNOWN，不重执行，约17:19统一原路径只读恢复。

root明确原剩余窗不足1795保守代理，禁止硬挤或延长运行中旧窗；批准CPU+源/计划push后，在**独立新namespace新1800+30**恢复。原任务真实终态和17:39:49旧截止不改，四患者/seed/采样/供给门/损失均不变。此刻新GPU未启动，`evidence/loading-metadata-failure.json`给不可变原证据。
