# 有上下文的实体二元问题槽：成本与隔离可行性

目标是判断 9B 固定候选问题槽是否值得作为 HGB 的新增实体判断来源。本阶段只做 CPU 来源、标签和实际分词统计；不跑神经模型，不读 F1 金标或原 dev，不评价新质量。

## 已经做过什么

严格 Biomed 上下文 CE 已做过：6 套 train16→16 候选、3 套 train32 CE，跨度/HPO 名称与 ±180 字符上下文，给 HGB 增一列，outer0 Score .714807668→.701372111，四项下降。另序列路线的 HPO 上下文重排也曾在 CRF 池下降。因此新假设只能是 **9B 指令与较长上下文、固定二元槽这一配方是否优于小 CE**，不能把“首次加入上下文”当作依据。公开 PhenoBERT 不看文内上下文，两个控制均负，已关闭。

检索正在验证的固定槽有可复用的数学接口：无答案输入，读取每个问题末端的实际 token 位置，冻结 LM head 的 token 15/16 两列，FP32 二类 CE。其标签是患者归属，本任务标签是实体跨度/ID 是否匹配，不能把检索质量当实体质量。现有训练 loader 写死患者 schema，新实体入口需独立适配，不能直接调用或修改活跃源。

## 严格划分与全池统计

新 heldout=F1，train32=F2+F3。训练候选用 F3→F2 与 F2→F3；F1 候选用 train32→F1。现有逐文 OOF 四折 HGB 候选不自动满足这个新外层隔离：用于 F2/F3 的生成器可能见过 F1。此风险在准备时拦住，没有将其投入新训练，也不据此否定旧四折结果。

从旧严格 CE 的原始候选重建完整语义+本体池，不用曾按金标 1:3 平衡的训练子样本。此可行性池不含 CRF-only/新增公开来源，不能冒充当前最优 HGB 全池。先冻结全部输入和候选，再仅从 train32 派生二元目标；目标沿旧 HGB 的 exact span/ID subset 与 gold NIL wildcard，非纯本体等价标签。

唯一 CPU 普查耗 **43.94 秒**，上限 120 秒：

| 范围 | 候选 | 提示 | 实际 token | 最长 |
|---|---:|---:|---:|---:|
| train32 | 28,933 | 1,747 | 3,624,190 | 4,792 |
| F1 input-only | 15,827 | 934 | 1,954,647 | 4,702 |

训练 1,721 正 / 27,212 负，1,078 条提示全负。每条最多32候选，同 passage/512字符起点分组，提供组前后500字符和指定HPO名称/定义；未按概率/标签丢候选，没有截断。实际 tokenizer.json 的简化两消息模板在既有124条 AutoTokenizer 实际槽记录上逐 token/位置/渲染 SHA 全一致；所有新坐标、槽边界、token roundtrip 通过。

按检索最长6777 token 单步13.123秒线性外推，原全池2epoch约 **14,036秒/3.90小时**；这是估计，不是实体训练实测。检索中途第一轮实际吞吐折算约2.95小时，仍远超最小控制预算。因此不建议启动全池训练。最长训练行 `11765409:47:0:0` 有32槽、全负，按输入长度选择，不能换成金标挑出的“好样本”。若后续单步诊断获准，还需另用合成正负 logit 验证双方向梯度；丢弃全部 probe 状态，不推断质量。

## 唯一门控缩小普查（完成）

根线程追加授权：仅 train32 标签和严格下层资源，以旧语义+本体 `calibrate.features`、HGB160/15叶/最小叶35/L2=5/lr=.06/seed20261023/no early stop 新 fit 一次；固定宽松 **p≥.1**，不扫 cut。新模型只用来缩小候选，不把 gate 分数喂给9B。训练32上的筛选是 **resubstitution selection**，不是 OOF；F1 仅无标签过滤。

新 fit+推断+成本普查上限60秒，保存真实模型并回载核概率；统计全部训练正例保留比例只作训练诊断。不重用见过F1的四折概率，不冒充当前97列公开HGB。计划 `gate-p01/plan.json` 已在运行前冻结。若成本仍过高或资源链不完整，则停下缩小方案，不据金标调整阈值。

完整产物：`/workspace/patientphex-runs/round2-20261023/sequence_crf/entity-slot-feasibility/`；小索引见 [证据](../evidence/round2/entity-slot-feasibility.json)。`census.py` 是实际运行源码，`gate_census.py` 是新增成本普查入口；引用旧源码小快照只读保留。既有当前best参数9,475,880,164，若共享同一9B底座再加独立43,278,336 LoRA，合计9,519,158,500；此仅资源预算，不解释官方ensemble口径。

固定门控 CPU 新fit **2.048秒**，fit+推断+分词+回载总 **17.159秒**。77自然列，模型回载两集合全部概率exact。训练候选缩至 **2,430（1,645正/785负）**，训练正例保留95.584%；637提示/460,069tokens/最长2,418。F1仅input缩至1,589候选/398提示/295,155tokens/最长2,810，所有文献仍有候选。未知F1保留正例覆盖，不能用训练95.6%代替它。

根已条件授权A100 GPU2一次**总120秒probe**，正式训练未获准：最短259token/1正槽、最长train2418token/20槽15正，各一步非零lr更新；最长F1 2810token/23槽只forward检查shape/finite，不保存分数。行按输入长度选，非标签筛选。固定源完整复制至自有目录，复用未改的load/slot/loss数学；实际远端AutoTokenizer再核3条输入，native全logit/指定slot对照、LoRA各family真实更新、LMhead不变。115秒主cap+5秒TERM/KILL，全部状态丢弃。未来若训练须fresh2epoch/r16/alpha32/.05、batch1/accum8、lr1e-4等事前配方，依据实测重新定总预算，不能从probe自动续训。[probe计划](../evidence/round2/entity-slot-gate-probe-plan.json)。

## 实际短检查完成，正式训练待预算决定

同一公开9B fresh加载、actual15模型文件完整SHA通过，A100 Torch2.6/TF5.20.2/PEFT.21.2；唯一进程 `proc_76a0782e-90c8-44e5-87bb-84750644eea9` 在 **84.067秒** 内完成（含校验和加载），没有保存adapter/optimizer。23:02:01.597Z fresh检查GPU2无进程。

- 最短训练行259tokens：一步 **2.545秒**；最长2418tokens：一步 **4.348秒**，峰分配 **20.972GB**。两步各LoRA三模块家族有有限非零梯度和真实更新，冻结head无梯度/字节不变。长步使用短步更新后的probe状态，不称fresh训练质量。
- 最短行native完整logit与指定slot完全一致，错用causal `s-1` 差2.375；3条实际TF tokenizer输入/位置与本地编码exact，合成正负梯度方向通过。
- 最长F1无标签输入2810tokens只前向 **.619秒**，形状和finite通过；没有保存概率，不评价质量。
- 按长步纯token外推2epoch约 **27.58分钟**；考虑两点截距的估计约 **62分钟**。短步含首次Adam分配、正式accum8又会减少optimizer调用，两个端点不能确定稳态吞吐，因此不沿用早先22–30分钟承诺，也不据此自动开训。

新数据独审 `3009d56c` PASS：原pool与H40三链、固定门控、1035条提示/全部token位置、翻转标签/污染metadata不进入输入逐项复核；未fit/GPU/读F1金标。实际probe增量审由验证代理收尾。完整恢复与成本边界见 [probe结果](../evidence/round2/entity-slot-probe-results.json)。当前best、原dev和官方提交均未改变。


## 动态微批工程探针（唯一150秒预算，尚未正式训练）

根线程授权新增独立探针，避免以两个冷/暖逐行端点外推整轮吞吐。保持seed+epoch的原shuffle顺序与8行optimizer组，组内按连续行贪心拆成`max_length × batch_rows ≤8192`微批；不跨组、不排序，末组真实5行。每行先对自身槽CE取均值，再除本optimizer组真实行数累积，不能按总槽数或微批数量平均。右padding/attention mask和原位置不变。

直接调用已注入LoRA的native text backbone，取每行实际槽hidden后以原冻结`lm_head.weight[[15,16]]`做F.linear；保留完整head及来源，不建新Parameter。CPU合同核数学/hidden梯度/非slot与padding零梯度、1/5/8行组和多种拆分的损失及梯度等价；实际GPU再核native full-vocabulary指定位置两列、独立非padding输入、padding token扰动，以及同短行实际LoRA_B各家族梯度。BF16事前容差logits atol.125/rtol.02、梯度relativeL2≤.05/cos≥.995，不把浮点或dropout draw顺序差异冒称训练轨迹bitwise等价。

仅29训练行按原epoch0组输入长度预选，不用标签选择：最短完整组warmup后再测一次、median padded成本组、最大padded成本组、真实尾5行；共5次真正optimizer更新。实际训练dropout仍.05，数学对照eval关闭dropout。全程含15文件SHA与加载145秒+TERM/KILL5秒，所有临时状态丢弃，无F1输入/标签/质量。新目录`entity-slot-feasibility/microbatch-probe-v1`，旧probe不改。详见[evidence](../evidence/round2/entity-slot-microbatch-probe-plan.json)。正式训练仍待root根据实测决定。

首轮probe统一独审已完成：`validation/entity-slot-probe-review/review.json` SHA `c0a286eb227d3314c0949ca633dc88d0a886c27215c7c04b399d87035da4f139`，复用数据审3009d56c，不重复旧模型或数据检查。


### 微批实际终点

新probe `proc_f8e9909c-8ee3-4360-a2db-af81a521f4fe` 23:07:08.213Z开始，**115.957秒**完整结束，含15文件SHA/加载/数值和梯度检查；23:09:12.212Z目标GPU2已空。完整终点SHA`9c0ef0d3`，实际源`16d45ce6`。不保存模型或optimizer，无F1输入/标签/质量。

| 实际组 | 真实／padded tokens | 微批行数 | 秒 |
|---|---:|---|---:|
| 短组warmup | 3828／5232 | 8 | 4.230 |
| 暖短组 | 3828／5232 | 8 | 4.085 |
| 暖median组 | 5049／8731 | 7+1 | 8.568 |
| 暖maxpad组 | 8050／14860 | 4+4 | 15.497 |
| 暖尾组 | 3589／6115 | 5 | 5.568 |

峰allocated **26.786GB** /reserved32.204GB。native同batch两列与原head两行F.linear差0、padding token毒化差0；batch与unpadding逐行最大差**.25**，满足事前atol.125+rtol.02而非bitwise相同。实际LoRA_B三家族native/compact梯度relativeL2为.008759–.011753、cos≥.999931，原head不变；五组真正非零lr更新均通过。CPU独审`25d81432`确认原8行组/尾5/逐行mean和无新head；actual增量独审继续。

完整2epoch计数160 optimizer组、291微批、920138真实tokens/1458446paddedtokens。仅按四个暖组成本，两种透明代理估计更新阶段 **1407.54–1429.45秒**（约23.5–23.8分钟），再建议给hash/load/save约150秒余量；**这是预算代理不是实测整轮，也不是硬保证**，微批形状和动态kernel仍有影响。不再用旧冷短行线性截距62分钟估计。保持原目标和顺序，但batch改变dropout RNG消耗/浮点归约，应记新工程变体，不能声称等价复刻旧batch1轨迹。完整可重算成本与SHA见[结果](../evidence/round2/entity-slot-microbatch-probe-results.json)。是否正式训练仍由root决定，probe状态全部丢弃。


## 正式唯一fresh实体32二元训练（已授权，启动前冻结）

root在微批实测后授权原门控train32的637提示/2430候选，fresh公开9B、独立LoRA43,278,336/r16/alpha32/.05，BF16底座、FP32 adapter与AdamW、lr1e-4/wd.01/clip1，**固定2ep/160 optimizer calls/warmup8**。epoch seed20261023+epoch shuffle，原8行组/尾5及组内8192pad动态微批不变；每行mean再组mean、原冻结0/1 head两行投影不变。公开model、slot与microbatch实际接口全部复用，绝不续probe。

预算训练2200秒+TERM/KILL30秒；每组前固定留180秒保存余量+40秒下组余量，不足则保存partial并禁止推断。保存真实epoch1/epoch2 adapter、Adam/scheduler/RNG，epoch1仅恢复资源，**只评第2轮**。actual成本代理1407–1429秒更新+建议150秒加载保存，不是硬保证。A100预查空卡、持久盘至少4GiB后启动。微批actual独审`0b0c7cb2`已PASS，不重probe。

终点只在同一F1纯输入398提示/1589机械候选上，载最终适配器一次，用disable_adapter得到公开base，再启用final；原head双列、同右padding微批/同原顺序、固定**p≥.5**。双臂推断共享600+30秒，保存每一原key的真实两logits/概率，无生成/无搜索。两概率和两完整预测共同freeze后唯一F1实体质量检查，**不原dev**。

装配沿旧calibrate的`(-p,-length,offset)`稳定排序（完全并列沿冻结gated-prompt候选顺序），greedy拒任何重叠；相接允许，原candidate ID/compound/`-1`不改，统一type=Phenotype/note=None，无额外否定/NIL门。仅替换entities，原`7ef8b73d…`F1语义+本体baseline全部非entity对象逐项不变，不重新分配患者。5个输入边界CPU检查已过。当前**77列gate候选家族不含当前best97列公开多源完整池**；public-base对final的提升不等于超过现有best。候选F3→F2/F2→F3及train32→F1隔离、train32 resub gate偏差保持披露。完整计划与source恢复见[formal计划](../evidence/round2/entity-slot-formal-plan.json)。

正式入口增量独审 `eca2f72e` PASS：五个decoder合同与合成398行/1589键双装配、非entity对象保全、缺final/错key拒绝，以及实际推断前计划→training manifest→epoch002闭环通过。此仅源码/CPU合同；正式训练及真实预测质量尚未完成。


### 正式终点及独立复现（完成）

实际训练 `proc_3fc5df9d` 完整160次optimizer调用（warmup首调用lr0，159次非零lr）、291微批、fresh2epoch耗 **1460.803秒**；首/次epoch含各自保存690.160/695.909秒。没有overflow skip/partial，峰allocated27.035GB、reserved31.990GB。真实epoch2 adapter173,181,072B，SHA `0f238b049009f9f4ccb5f6f69d9b431a012c213bf1cc5b9ece2848f056a3e230`；Adam/scheduler/RNG状态346,666,706B，SHA `c126030809025af4fb4a083652e4caa93cbaafae5669ec63957ce0d5b411d757`。两轮均留A100持久自有目录，local只回收小manifest；没有宣称resume数值轨迹已测。

原生配对 `proc_21ce1665` 在同398提示/1589键完成，公开base前向67.197秒、final86.727秒，总230.765秒含校验加载；两次实际adapter文件SHA闭环通过。23:43:03.639Z GPU2已空。MCP API只报告running=false，没有exit_code字段；本地CPU装配和评估实际exit0。23:43:18.635Z两fullpred统一freeze（`bc73e942`）后，23:43:21Z完成唯一F1评价。

| 固定同池方案 | 实体数 | men F1 | doc F1 | Score |
|---|---:|---:|---:|---:|
| 公开base | 666 | .503219416 | .604003481 | .561134423 |
| 固定epoch2 | 1118 | .701740186 | .762190548 | .650311382 |

mention TP/FP/FN：508/158/845→867/251/486；document：347/57/398→508/80/237。关系437项原样，micro .593521421/macro .543793373完全不变；两臂format/HPO0错0警。两实体F1均值.553611448→.731965367，说明有限候选上下文监督优于公开base，但没有同本次候选/隔离链的现成HGB质量对照时，**不能据此说胜过77列HGB或当前97列公开多源best**。本次只改Task1，不原dev，不扫阈值或epoch。

独立终审 `f8f50357` PASS：真实训练160calls/两checkpoint预算链、两398×1589 logits→manual softmax、全完整预测byteexact、两v2/format全字段及全部非entity关系对象一致。复用已审data3009/microbatch0b0/entryeca2，不重GPU。完整实际源、概率、模型及状态恢复指针见[终点摘要](../evidence/round2/entity-slot-formal-results.json)。下一仅做严格同F1 HGB资源/可比性核查，待root决定，不自动扩大训练或迁移dev。
