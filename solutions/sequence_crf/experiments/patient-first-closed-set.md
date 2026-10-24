# 闭集患者优先生成：单步通过，正式方案待启动

真实目标是改变同一关系矩阵的自回归因子化，检验能否改善患者集合判断。约束是仅复用旧 strict train32 标签与 F1 纯输入；本阶段成功标准为全部关系双逆精确、来源隔离、实际分词/掩码和严格 schema 成立。**本阶段无训练、GPU、权重加载、F1 targets、原 dev gold 或新质量分数。**

## 固定改动

原 `CID -> [patient_id]` 改为 `patient_id -> [CID]`，每个给定患者都有键，空列表保留。只替换 system 前两句任务/输出 schema；后续临床判断规则、整个 user 文本、全局患者锚点、候选/HPO 映射、患者顺序、候选顺序、row ID 和行顺序完全保持。禁止输出未列 CID 或 HPO 编号。

训练标签使用原 raw-HPO 口径，不作 alt-ID 规范化。`transform`/`inverse` 对全部124行的完整对象及 compact answer 字节双逆一致；594个正 pair、1912个负 pair 不变。新输出总346个患者键，106个空值；旧880个CID键中396个空值，7条整行全空保持。该统计是序列化结构，不代表标签或质量变化。

## 实际结果

2026-10-24 实际 CPU **4.070265秒**，外层退出码0，固定 tokenizers 0.23.1 / Jinja2 3.1.6，未导入 torch/transformers/peft，CUDA隐藏。

| 项目 | 原方向 | 患者优先 |
|---|---:|---:|
| train 行/文献 | 124 / 32 | 124 / 32 |
| train 全部 token | 402,594 | 400,614 |
| train prompt token | 395,424 | 395,920 |
| train supervised token（含 EOS） | 7,170 | 4,694 |
| train 全长 min/median/max | 591 / 3311 / 5523 | 593 / 3298 / 5556 |
| train label token min/median/max | 9 / 59 / 116 | 7 / 31 / 135 |
| F1 纯输入总 token | 250,696 | 251,016 |
| F1 输入最大 token | 5691 | 5695 |

所有 train/F1 prompt 都恰好多4个 token。训练最长行仍是 `11235872:relations:8`，F1最长仍是 `6063015:relations:0`；没有截断。新 target 更短来自输出因子化，不能声称监督 token 数相同，也不能把token下降直接解释为更快或更有效。

复用旧实际 helper 的 `encode` 函数：对完整聊天字符串分词，验证 prefix IDs 精确前缀、assistant-only mask、唯一受监督 EOS 和 causal label 存在。**旧124行所有已保存 native 审计字段**（full/prefix/label长度、render SHA、input-ID SHA、label SHA）均精确；F1旧80行的 render SHA/长度与既有逆序准备审计匹配。新方向在真实 Transformers 模型运行环境上的 GPU 前后向/数值/耗时尚未测量，不宣称已过。

F1文件只有 `row_id/pmc_id/code_map/patient_ids/messages`，80行/16文献/39患者/587候选/1557 pair，没有 answer。只变换输入 schema，不读取任何 F1 targets 或响应。

## 隔离与来源

旧训练文件 `cb7f67e1…`、准备清单 `356109e9…`、native token审计 `46b005ef…`、实际旧formal2ep run manifest `5535d33a…` 均在运行前核完整SHA。两条候选生成链 F2→F3 / F3→F2 的 generator/target各16文献、互斥并合为训练32；F0/F1排除，训练与F1输入文献交集0。沿用已审候选来源而不重新生成或从label补候选；没有读取祖先manifest中指向的F1 gold。

tokenizer `5f9e4d49…`、template `a4aee8af…` 来自既有固定 Qwen3.5-9B verified-small。未下载模型。全部实际文件路径、大小、SHA在 `plan.json` 与 `cpu-v1/summary.json`；旧资源仍原位只读。

## 有区分力的检查

5类有效合同：P1/P10和C001/C0010精确区分；完整转置/逆转；全空值；空候选集；JSON键顺序不改变语义。

20类拒绝合同：重复/缺失/未知患者键、未知或重复CID值、null/数字/bool/string/nested错误值、markdown、尾随第二对象、NaN、把旧schema误当新schema、重复患者或CID白名单、空患者白名单、旧schema重复CID键/未知或重复患者值。新 parser 只接受一个严格 JSON 对象；实际字符串控制未靠宽松字符串搜索恢复。

## 可执行复现与交接

```bash
env CUDA_VISIBLE_DEVICES= TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/workspace/patientphex-runs/round2-20261023/research/thinking-budget-forcing-readiness/cpu-libs \
  timeout --signal=TERM --kill-after=15s 195s python \
  /workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set/prepare.py \
  --plan /workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set/plan.json \
  --output /workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set/cpu-replay-UNIQUE
```

脚本要求新 output 目录且用独占写入；已完成 `cpu-v1` 不可覆盖。保留的全量新数据：`cpu-v1/train124.patient-first.jsonl` 和 `cpu-v1/F1.input80.patient-first.jsonl`。源码为 `patient_first.py`/`prepare.py`，小证据与全量数据的独立保存边界见 `handoff-manifest.json`，供 sequence owner 选择性纳入其分支。

后续 trainer **不能直接绕过旧SHA guard**：应以新数据manifest绑定旧来源与本次转置，沿用原2epoch/32调用/每行CE归一/AdamW设置和原encode；只按明确的新数据schema调整来源校验。推理须用新患者键parser，再反投影成旧association，不得使用旧CID-key parser。当前源码已经提供这两个方向的严格纯函数，尚无GPU runner。

本阶段不改变loss权重、不开始训练、不自动扩大到F0或原dev。正式训练预算仍待真实新最长行成本；原124×2约1149秒仅是旧运行参照。若最终生产采用新adapter替换旧同形状adapter，参数不增加；若与NCR一起保留两个adapter并集，总量会超过10B（见先前 score-gap-review 报告）。

## sequence owner接管与唯一最长行探针

已逐文件核验handoff全部13件实际SHA，将两源码原字节接管到`solutions/sequence_crf/patient_first/`；大训练/F1输入与逐行audits保持外部。独立CPU数据审 d50c850e 已通过，25合成不重复。

主线程已授权一次A100 GPU2最长5556/135监督真实更新，180秒+TERM30，fresh public9B+全hybrid LoRA r16/alpha32/dropout.05，原生BF16/SDPA/nonreentrant checkpoint/native完整assistant CE。仅adapter进入AdamW，lr1e-4/wd.01/clip1；这是明确非零学习率的单步诊断，未执行正式2轮的warmup。初始LoRA B为0时A首步梯度可为0；检查有限loss/全部梯度、三family正梯度和各一个小B矩阵真实变化，不复制9B base。状态退出即丢弃，不保存checkpoint、不推断、不质量评分。

新data_contract独立绑定新train SHA、原来源与逐124行实际tokens，严格124/max5556/total400614/supervised4694；只复用旧底层encode。原旧trainer的7170/402594/5523 guard保持原样，未绕过。虽然仍用有效assistant token均值CE，输出顺序和自回归分解改变会改变token级优化目标，不能称原数值目标等同。

参考已审A100旧5523 selected-weighted一步12.1805秒/峰23.8591GB；更接近算子的旧A800 native2ep峰36.6334GB、reserved81.1095GB（多形状训练池）。两者都不保证本单5556在A10040GB的实际峰/耗时，OOM/数值/预算首失败就停，不重试更换attention/loss/截断。

**当前阻塞**：A100 runtime只读核验立即返回LUNA_OFFLINE，未取得新runtime执行结果；ontology NCR成本probe亦尚未启动，root统一恢复连接。当前仅本地source/manifest/AST与静态增量审3275b461完成；真实nativeCPU绑定与GPU更新均未执行。等待ontology成功的实际只读roundtrip后恢复同一次CPU检查，先让其NCR短探针完成释放，再fresh查GPU空闲启动本单步；不凭mount online认定恢复。

外部`probe-preparation/`有resource/data-plan/source proposal/payload及MCP故障收据。恢复入口：`patient_first/deploy_cpu.py deploy --directory <probe-preparation>`，完成后`recover`回收三CPU文件；若工具返回unknown，先检查已落盘文件，禁止重复盲发同一部署/CPU动作。源/计划推送、实际CPU收据与最终GPU run-plan冻结后再使用`operations.py start`的单次intent入口。GPU启动前无需新增用户确认，但具体阻断必须先处理。

### 连接恢复后的部署中断（仍无CPU/GPU模型执行）

ontology于01:52:11实际run_command恢复成功后，本方案仅在自有目录部署。目录创建被controller观察为rc0；唯一254466B `upload.part000` 的write_file返回UNKNOWN_OUTCOME，随后只做一次该part实际SHA读取也返回UNKNOWN_OUTCOME。**不能记成part不存在或传输成功。** 没有解包、native runtime/data CPU命令、GPU start-intent或GPUjob；已停新MCP，等待root统一恢复。ontology的NCR也尚无GPU作业，不存在本方案占卡等待。

源码53d485c已推；本地zip、18文件清单、intent及两次未知收据完整，见[精确分段状态](../evidence/round2/patient-first-probe-deployment-state.json)。恢复后只核已有part实际SHA；若与zip相同，解包既有字节并逐成员核验/compile，再执行尚未执行的CPU步骤，不重新上传或调用`deploy`从头创建目录。`deploy_cpu.py recover`仅用于三个CPU结果已经落盘后的回收，不能替代本次upload-part库存检查。


### 原生 tokenizer 的 CPU 导入检查修复

01:58 实际恢复后，原 part 被确证不存在且目录为空；仅补传原冻结 archive（cd50df2c，254466B），18成员 SHA 与实际 Python compile 通过。native runtime 首次实际通过：Torch 2.6.0+cu124、Transformers 5.20.2、PEFT 0.21.2、原生 GDN/无 FLA，CUDA隐藏且未初始化。

首次 native data CPU 在构造 tokenizer 后、124行编码前因“禁止 Torch 出现在 sys.modules”过严 guard 退出（13.098秒 roundtrip，rc1，无产物目录），不能记为模型或数据失败。单独实际导入检查证明只有 Torch 被携带导入，`USE_TORCH=0`仍在，CUDA available/initialized均false，无权重或forward。窄修为导入前必须隐藏CUDA、后续实际未初始化、禁止其他后端；记录真实Torch导入，不再声称所有后端未导入或CUDA查询次数为0。六输入SHA、逐行token/label合同、最长5556、训练loss和GPU180+30全部不变。原源码/计划/失败保存在原archive及53d提交；远端新文件使用`data_contract_v2.py`与`data-plan-v2.json`，不覆盖首次失败入口。[修复与来源](../evidence/round2/patient-first-CPU-import-guard-repair.json)。


### 实际 native CPU 与唯一 GPU 计划冻结

修复后的 CPU 子进程7.189640秒、rc0（合同内部6.284941秒）；真实TF5.20.2 tokenizer对124行所有新token/label哈希及32文献来源exact，总400614/监督4694，最长5556/5421/135。Torch实际导入，CUDA仍隐藏/未初始化，无模型权重或forward；没有重跑旧25例。三件原始收据/编码已按实际SHA回收，并保持原字节flat复制到远端根目录。选中编码9c84e688（108350B），CPU收据2c5d8254（8020B），runtime689d3317（905B）。[CPU收据](../evidence/round2/patient-first-native-CPU-receipt.json)、[真实环境](../evidence/round2/patient-first-native-runtime.json)。

[最终run-plan](../evidence/round2/patient-first-probe-run-plan.json)固定原23成员SHA、actual CPU收据及选中编码，明确fresh state丢弃、180+TERM30、仅一次非零LR1e-4/native full CE真实更新；无推断/质量/正式训练。NCR实际01:58:17释放GPU2，启动仍由原operations核fresh空卡/写唯一intent。先完成并回报峰值/耗时，再由主线程安排后续训练；本体修复后的NCR在本probe实际release后接卡。


02:00:59.317Z 已fresh核A100 GPU2 0MiB/无目标进程后启动唯一`proc_9d262dcf-1234-47db-88d0-ec6c1a941d8f`、PID6498，源码检查点0a87abc。原180+TERM30，近似TERM02:03:59/KILL02:04:29，actual deadline由远端driver-started记录。只运行本最长行一步，结果未知；终点实际release后将卡交NCR，不能重复start。[启动/恢复](../evidence/round2/patient-first-probe-launch.json)。


### 唯一最长行更新终点：通过，仅限实际更新与成本

原proc_9d262dcf完成，native79.031891秒/wrapper80.192565秒、rc0；原180+30内，无重试。单步13.238732秒，loss .268331736、clip前norm2.70594263，三族梯度均有限非零；248个B梯度非零、A首梯度全零与fresh B初始化0一致，三小B探针均真实变化，base无梯度、优化器仅adapter。实际base8,953,803,264 + LoRA43,278,336 =8,997,081,600，加载无missing/unexpected/mismatch。没有保存checkpoint/推断/质量，状态随子进程退出丢弃。

峰allocated36,200,963,072B（33.715GiB）、reserved40,955,281,408B（38.143GiB），结束前free793,051,136B（.739GiB）。02:02:19.724Z wrapper实核目标卡无进程、0/40960MiB；已通知ontology接其NCR，不开正式训练。8个终点原件/37项manifest已回收全SHA，stdout/stderr和完整模型库存外存。[结果](../evidence/round2/patient-first-probe-results.json)、[清单](../evidence/round2/patient-first-probe-terminal-manifest.json)。

按801228训练tokens线性缩放单步约1909秒（31.8min）；若248微批均按该最长单步计约3283秒（54.7min），另有本次约65.8秒加载/校验等。两者只是成本代理：正式accum8/32次Adam，与probe每步Adam/诊断不同，长度、算子、I/O和负载会变，不能当实际完成时间或严格上界。若主线程选择，建议训练硬3900+30秒、后续完整80推断另600+30秒；此处没有启动授权扩大。A100首步forward尚无Adam state，正式后续常驻Adam/累积梯度及变化shape内存池可能更高，单次最长通过不保证2ep显存；优先等80GiB A800空闲，不自行改loss/attention或截断。仍无质量分数或达标推断。


### 统一终审与正式方案准备（未启动）

单步终审`3f22cbfc`通过：37refs/23部署/8终点与15模型元数据、nativeCPU窄修、真实更新/数值/退出/释放均独核，没有新NN或重跑25例。原data CPU过严guard失败仍保留；正式阶段沿用新的实际CPU收据，不再把Torch导入等同GPU或NN执行。

主线程确定先等A800 GPU2现warm最终释放；fresh全卡survey无其他空卡，GPU0外部PID不共享或停止。独立输出`/dev/shm/patientphex-patient-first-1045/formal2ep`，只读复用reference SHM base/runtime；启动前还要实际空卡与SHM至少2GiB余量，.100持久空间至少2GiB，源/数据/基座实际SHA检查。当前仅准备，**没有正式launch授权或job**。A100旧显式配置没启用allocator选项，历史继承变量未保存；当前hiddenCUDA CPU实查Torch2.6/native、两allocator env均null，无新override。没有把当前配置回推旧运行，也不新增expandable_segments试探。

正式训练源`a3ec17f5`从旧`08fc142c`最小改编：原fresh模型加载到两轮、32 optimizer calls、2 warmup、bs1/acc8、seed20261023/r16/alpha32/dropout.05/BF16/SDPA/nonreentrant/native逐行CE，以及epoch adapter+Adam/scheduler/Python/NumPy/Torch/CUDA RNG保存/恢复的整段源码逐字不变。新manifest明确区别旧provenance的cb7f train与新14af转置train；每行新schema/10字段token审计、400614/4694/5556和actualCPU combined SHA均硬绑定。唯一新增CPU源接线0.088秒实际返回124/32train/32excluded隔离，两个epoch shuffle逐字同旧、每轮15组8行+末组4行；H44反例只排除同前缀ack文件，不冒称完整GPU恢复等价。旧meta/causal CPU模式没有单独重跑。训练硬3900+TERM30；首OOM/数值/预算失败停，保留完整epoch，不减输入、不改loss、不自动重启或推断partial。

最终推断仅epoch2/32calls、同原80输入和旧非thinking greedy设置14000/1600（不另改cap）。新input总251016/max5695，输入级synthetic全正compact响应参考最长303token；这是具体序列长度，不是任意空白或模型输出的上界。新增1有效+6拒绝EOS/身份合同，旧25 schema检查沿用；真实generated IDs/单终止EOS、完整PID键、合法CID及无重复、整串JSON必须全过，反向转回CID→PID后装配，禁止fallback。完整80与fullpred先freeze，再唯一F1 v2；原1189实体/非association字段完全不变，旧SFT32指标直接复用。保留条件固定`delta(micF1+macF1)>0 and delta(macF1)>=0`，否则负关闭，不并集、不扫参数/轮数、不原dev。推断另600+TERM30，CPU装配/评分各30秒。

最终完整checkpoint计划由已核HTTP方式在独立180+TERM10内从A800自有SHM导出到`.100/sequence-round2/patient-first-formal/checkpoints/epoch-002`，只复制白名单adapter/Adam/RNG及小来源文件；目标fsync后重开完整SHA，再独立读回校验；动态端口、源进程记录先于start，未知结果查原进程，确认文件固定`export-ack-epoch-002.json`。新HTTP源使用兼容3.8的resolved-parent包含检查，避开H33。成功回收只停止本次server；连接未知时原远端180+10仍有界。最终导出实际成功前不进入推断。适配源码/恢复CLI已准备，未部署或发送权重。

运行/恢复入口为外部`formal-preparation/`：[准备总清单](../evidence/round2/patient-first-formal-readiness.json)，`formal_operations.py deploy/start/collect --phase train|infer`负责唯一部署与phase intent，`export_operations.py prepare/start-server/start-fetch/collect/stop-server`负责CPU持久导出，`finish.py assemble`先freeze，随后才`evaluate`。实际runplan/文件SHA见清单。准备脚本与全部嵌入远端Python模板均compile检查；当前GPU/磁盘fresh preflight及新实际运行仍待资源释放和主线程安排，不把计划写成已执行结果。


02:19:56只读实际存储核验通过（[收据](../evidence/round2/patient-first-formal-storage-readiness.json)）：A800 uid1045，SHM可用248.569GB、自有输出/源码目录均未创建，原reference model目录存在非symlink；project磁盘仅177.066MB，所以只放约几MiB source/小日志，不落权重或state。.100 uid1006，持久磁盘可用443.631GB，自有checkpoints/tools尚不存在。未查询/操作GPU、未重复权重SHA、未mkdir/部署/启动。正式source原入口仍在GPU加载前核完整15文件SHA，卡和SHM余量在真正start时重新核；此存储状态不代表warm资源已释放。


### 正式完整执行授权与部署

主线程已授权完整fresh2ep→持久epoch2→固定80→唯一F1，不再等待逐步许可。原23件正式payload已实际部署/逐SHA+compile通过，原archive92834857及deployment-intent原样保存。独立新HTTP/fsync合成合同ac4d5209实际0.466s完成11件输运、目标fsync/回读SHA/ACK及4类拒绝，沿用该证据不重复旧Range检查。按审查建议只为3控制器save补flush/file fsync/dir fsync与必要os import，实际函数落盘/拒绝覆盖通过；原源码/计划快照位于formal-preparation/before-controller-fsync，数值训练/推断/数据不变，新readiness d5dfa72f。远端只窄补3源+2plan，不重部署原数据。

检索已给最终GPU释放：02:25:04.529616Z A800 GPU2 f516无进程/81156MiB，release856903f5，原camp无后继。尚未运行本路线实际A800 CPU/GPU；接下来CPU隐藏CUDA核124编码/原生后端、fresh资源与模型SHA后直接按既定3900+30启动，不把收到释放消息当本次GPU检查。


02:27:12.289Z actual A800 CPU完成21.456s：TF5.20.2/PEFT0.21.2/Torch2.7.1+cu126，native modeling b4b17fe0及两个GDN binding原生、无FLA；hiddenCUDA未初始化、无权重load/forward。真实正式trainer.verify_sources_and_data/encode_all对124行400614/4694/max5556及combined8d69e592完全匹配；15个只读SHM模型文件均完整SHA。为ID隔离实际读取了包含F1文献的excluded32纯输入，未读F1 targets/gold/单独提示或预测；原回执F1_data_or_gold_reads=0字段过宽，保留原件并附scope clarification，不重跑CPU。两allocator env实际null。3源+2plan窄补后两计划全部21成员SHA精确，旧件远端独立快照；.100 fetcher部署完成SHA/compile。详见[实际入口](../evidence/round2/patient-first-formal-actual-entry.json)。下一操作是原授权fresh GPU核验与唯一训练start，不重复CPU或改变配方。


启动前fresh资源通过，但start_process返回明确PATH_OUTSIDE_MOUNT，没有managed job；原intent已落盘。故障来自MCP start的cwd命名空间要求相对挂载，而控制器用了真实绝对路径。保存原intent/source/plan在before-cwd-fix，只改控制器cwd为同一目录的挂载内相对路径；命令、训练数值、输入和3900+30保持。恢复必须先实际查控制目录/输出/driver进程/GPU，确认原start没有运行后，按独立恢复intent发送一次；不能直接重调start覆盖原证据。


02:29:03.366900Z 路径恢复后真实启动proc_d9d0823d-54a7-47a2-be01-87b5aca2ade0/PID52420，A800 GPU2 f516。恢复前actual control空、训练输出不存在、匹配driver进程0、卡1MiB无process，窄补三件后两phase全部SHA通过。原被拒start不是模型尝试；独立恢复intent与RPC/job保留。源码439ff981已push核远端；硬3900+30，近似TERM03:34:03/KILL03:34:33。当前无训练/质量结论，只监控原job并按冻结训练→持久epoch2→最终80→唯一F1收尾，不再start。[启动清单](../evidence/round2/patient-first-formal-launch.json)。


唯一正式入口独审71d71097 PASS（44冻结refs）：包括actual A800 CPU、三控制器fsync、新HTTP合成合同、F1纯输入scope澄清、cwd窄修和唯一真实job；原train/infer数值与完整数据隔离绑定通过。审查未新增CPU/NN，也不代表正式训练、最终80或质量已完成；终点仍按上述原任务收尾。[入口独审](../evidence/round2/patient-first-formal-entry-review.json)。


02:40:44实际监控确认epoch1完整，16/32 optimizer calls，epoch557.7481575s、平均accumulation-group训练loss .2666083614；epoch-001已完整保存，继续同一进程第二轮。step15峰allocated36.7366GB，nvidia-smi缓存/进程占用69559/81920MiB，原80GiB仍有余量，无OOM/数值异常。首非零step2/lr5e-5、norm6.57372、三族B真变化如原warmup；未评epoch1，且此时仅SHM检查点，尚无持久导出声明。[首轮进度](../evidence/round2/patient-first-formal-epoch1-progress.json)。
