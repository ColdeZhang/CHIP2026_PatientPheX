# 工作台账

分支 explore/quote-set-completion，仅维护 solutions/quote_set_completion。**已完成并关闭**：F1 .702224686较B4+.000899309，但F0 .724106276较B4−.004089353（新增1TP9FP），未过双折正门；禁止启动条件full64训练。所有GPU已释放，唯一F0评价完成1v2/旧0；稳定入口 `N/F0-complete-review-entry.json` SHA caa6ed94，详见实验QSC-001。F0最小持久补包已82+1实际服务器核验，receipt33ca4ca5；当前只剩集中增量审与root唯一下载恢复。F1已独审且root真实回下载。

恢复：N为仓库外 research/quote-set-completion-v1。不得重复正式训练、F1/F0推断、规范化或评分；条件519/36 CPU产物只作未执行计划保留。最近源码/包装检查点2563dc2a已push，原dev最佳不变。

## 阶段历史

以下按发生顺序保留，当时的“尚未启动/授权”不代表当前状态。

目标：检验给定已报告表型时的条件补全，是否比独立重新生成更能回收遗漏患者关系。当前原dev最佳 .769235669942；本线不读其gold、不宣称预计达标。先复用原train32的87患者弱quote目标，构造两个互补known/remaining视图；原真实关系标签不变，词面代表弱锚点而非人工患者证据。原双采样F1正/F0负已在旧分支关闭，不能重复其采样共识。

公开来源已实际HTTP200缓存Self-Refine、Levenshtein Transformer、BART及内在自纠错的反证论文；只借成熟条件去噪/插入目标，不宣称复现其模型或公开收益可外推本赛。CPU计划plan.json先固定，完整供给/长度核后再给具体成本与单一验证方案。训练与GPU尚未获授权，不占A800。

CPU唯一准备已完成 3.613秒：87患者→174互补视图，每个原801quote恰监督一次，25空known/25空remaining；全部920222tokens、最长9307、最大监督307。输出仅仓库外，manifest5210e0f3。见evidence/readiness.json与proposal.json。该结果证明可构建，不证明补全有效；完整HF runtime重放/新warm-start训练接口/真实压力与held输入仍未做，无GPU授权。元数据汇总首次只有Python保留字SyntaxError、解析前0写入，改字段名后复用原CPU终点，未重CPU/tokenizer。

## v2 完整停止视图与原生合同

Root在首次GPU前授权固定第三视图：每原87患者加入full-known→[]，261行单轮/freshAdam、warm G32 d52、LR1e-5、33updates/warmup2/尾5。v1的174提案原件保留但不执行。v2首次loader误插入列表assert、before-tokenizer TypeError，修复另源后唯一localCPU5.105秒，261行1380333tokens/max9307/监督12406/max307；112空target中87为新增停止视图，不隐藏目标分布变化。

唯一远端CPU15:07:32→15:07:51，19.062秒rc0；15原件SHA回收c2ff0d45，261全部HF/native IDs/labels/rendering/EOS、known/target集合隔离及33-update toy完整状态接口通过。未载模型或质量。见evidence/native-v2.json。

下一唯一240+30工程：warm-d52/freshAdam，固定8次反向非零更新→3次累积→最长9307/187与最多监督5773/307，13backward，保原indexedCE/native/BF16；完整adapter/Adam/scheduler/所有RNG临时SHM序列化并重开校验后弃掉。正式训练草稿尚未运行；不能沿probe权重继续。源码/plan先推后fresh门启动。

原13步工程路径修正后15:13:27→15:15:28按原绝对截止TERM，120.847秒rc−15；已真实group8 Adam/三族参数变，未完成全部极端shape/保存，不能记PASS。首次beforemodel5.225秒失败及恢复partial全部保留；旧进程/卡已实际释放。70秒剩余门过于乐观，已纠正成本依据，不再复用。Root另授权独立180+30最小工程：仅冻结两极形状各FB合并真实Adam，并全state保存/重开验证，fresh父d52不沿partial；不重复13步或261CPU。

新独立两极工程15:16:30→15:17:33完整63.351秒PASS：2FB+1Adam，峰27.791GB/28.255GB，所有496梯度有限/三族B变，全adapter+Adam/scheduler/Python/NumPy/Torch/CUDA RNG序列化/fsync重开逻辑exact后原scratch删除；23原件1984c2c2。原failed10件b02与timeout41件998独立保留，不混终点。

正式尚未启动：预写max(两极token代理3012.506,同host完整epoch代理2567.219)×1.2+300=3915.007超过3600，CPU成本guard在生成formal计划前停止，证据d1d495cc，已报root；原partial9仅成本代理2561.377，不当覆盖证明。完整39 known-only输入109c已构建（328原Gaccepted,6空，211943tokens/max10854），纯本地native/HF实际待infer前绑定；不读heldgold、不评分。

Root在任何formal部署/启动前明确把唯一共享预算改为4200+TERM30，仍预留120秒ACK（模型绝对窗4080），不降低max代理×1.20+300公式；原3600源/失败d1d保留。实际完整trainer.verify_inputs CPU已通过261/33/尾5/所有ID/位置/来源，ac104d60；正式plan/controller b7a9bb2f，见training/frozen及evidence/formal-authorization.json。下一fresh卡/盘门后唯一启动，SHM完整state→.100逐SHA/fsync/reopenACK后才39推断。

唯一formal实际15:20:44.761823 start proc_ad68dbeb-c7e6-453a-821a-b0fac0d57ab9/driver13674，freshGPU2空、SHM215.866GB/.100393.918GB；4200共享campaign起15:20:43.644688、模型TERM16:28:43.644688，持久ACK最迟16:30:43.644688。detached/flock watcher PID/SID190459（ready0593adbf）唯一每180秒原inspect，健康时不另查。恢复入口training/controller/operations.py inspect --directory N/formal-v1；不可重复start。关键坑精确源见evidence/handoff-pointers.json。

后继完整39推断/必要Sap/唯一B4union源已准备：只有正式完整ACK后bind实际权重；原始生成IDs→raw→strict解析→固定train32/public41099规范化，所有合法新关系统一并入B4，不评单臂、不筛新增。all39 HF/native IDs在模型加载前核；后继1800+30、Sap180+30和唯一v2 30+1分别独立，实际计划等待前驱全量终点绑定。固定参考postprocess/fixed-references.json已锁，当前无held输出或质量。

15:28阶段交接：root明确要求保留原formal/watch并释放代理槽，root只读已有本地观察，完整ACK或实际故障后followup恢复本线。精确恢复协议见RESUME.md/evidence/live-formal.json；后继源已远端be0ab79d，全部GPU/NN/质量实际绑定仍待原训练终点，不停止或重启任何原作业。

原formal完整成功：261/33、2464.423秒，峰28.292GB allocated/32.617GB reserved；新adaptercd977a24/fullstated90b14fd，完整持久ACK4b486327。原watch正常结束；原GNU timeout HTTP wrapper rc0/85.837秒，自有driver和timeoutchild实际已退出。34原件collection3c799b44逐SHA核齐，无重训。后继39实际plan b89a8f45已绑定；按H69新增5分钟启动有效期，入口/子Popen/模型加载前核，独立1800+30运行钟不变，12个exact-source CPU边界检查通过，未增加模型/质量。

39唯一推断已16:08:29.878 start proc_4ac60fcc-feb9-4000-baec-d052cea8547a/driver36169/child36205，实际原TERM16:38:30.114/KILL16:39:00.114；admission截止16:13:28.846，全部39 HF/native输入核过、模型加载正常。只读watch沿已核full64入口仅36→39，下一次沿现有observe+180秒，不重复启动或加频。窄读helper仅核新后继schema无阻断即结束，未造新审查或重跑旧sample。

Root已预排必要Sap在.100 GPU0（teacher16:07真实释放），全39后fresh门直接唯一180+30；不占.3的adapted query或GPU1 BOUNDARY。H69只新增启动有效期，原Sap查询/driver剥离admission插入后AST逐同、12边界检查2152eeac通过，固定FP16/41099bank/全部query/policy不变；实际plan仍待完整queries绑定。

39原完整终态已123.479秒rc0，16:10:33.615实际释放A800GPU2。92原件ca965d2e全SHA回收；普通G逐item合同39 EOS/0cap/0abstain，33合法空数组、其余共42精确anchors/40词面。固定strict准备仅需28个public查询，未读gold/质量。query plan1e59d2a7，原policy与所有quotes保持，按root预排.100GPU0/fresh门唯一180+30。

F1唯一质量已完整：pred50bb61dc/141199B，freezebc8a9de1在先，唯一新v2/旧0 resultfb352642；Score .701325377212→.702224686258 (+.000899309)，新增2TP0FP，micro+.002391889/macro+.001205347，Task1不变。稳定entry a9044674/36466B已交统一终审，结果evidence/F1-result.json，细节唯一实验QSC-001。Root授权同cd977 F0全47/强B4 .728195628911/普通G32合法known，1800+30和必要Sap，负平关闭；新full64仍仅条件CPU，须F0严格正另预算。

F0准备完成dbf30e16：同cd977、原G32 acceptedknown204、16空known、全47/16纯输入296853tokens/max10262，低于实际F1max10854；train32/d52/cd977/normalizer完整祖先与F0文献互斥。新plan e3047615，同native greedy1600/1800+30/5min admission；唯一对照B4F0 .728195628911/343关系/947实体，固定并全completion，负平关闭。源/输入/基线绑定CPU通过，尚未F0 GPU/质量；不重旧probe。

- F0 唯一启动16:22:33在载模型前资源门失败：GPU2新PID46312/484MiB（target493MiB）；driver14768明确退出，无child/forward。原N/inference-F0-v1与H69意图保留，未重start/未碰外部PID；待root资源协调。F1补包203+1原成员正在.100持久部署。

- F0资源恢复预锁：原A8000forward明确终态保留；同cd977持久ACK改.100GPU0，f0_a30/source-contract记录全部机械差异。CopyF0实际max9935/峰20.98GB，新10262多327tokens，root明确允许按余量做完整47有界尝试，不冒称exactshape覆盖；1800+30/启动300秒不变。F1最小补包已203+1实际服务器SHA/fsync/reopen，根唯一下载另记。

- F0 A30实际16:28:43启动proc_bca573cc/driver2754362 child2754370；TERM16:58:43.887/KILL16:59:13.887，1800+30。只读watch见live-F0-A30。原47/known204/同cd977完整保持，备份/conditional519 CPU已分离；不得把full64标签用于本F0。

- F0 A30原47完整：223.082秒/47EOS/0cap/峰21.066GB；16:32:27真实释放GPU0。108原件逐SHA回收e27cb54d，原固定解析53anchors/47词面、必要Sap27，完整规范化后唯一B4union质量。F1实际服务器恢复bf565已root完成；条件full64仍仅CPU。

- F0最终固定端点已负：Score .724106275971，Δ−.004089352940；mic−.007296066/mac−.009061346，新增10=1TP9FP；全部947实体/47患者/343旧关系保持。原pred11622f97/freezefebf75c8后仅1v2/旧0，resulte51864d0。关闭QSC，不启full64，未重试格式失败或按错误删答案。

- F0唯一最小补包975394B/b1c95a2c已15分块完整持久，82内容+1内嵌manifest逐SHA/fsync/reopen，receipt33ca4ca5/30181B；未重复F1或weights。包准确标注F0独审待root集中安排，根客户端下载另待；稳定入口caa6ed94保持。
