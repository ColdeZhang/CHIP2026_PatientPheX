# 第八阶段：官方反馈后的并行扩展

用户回传 2026-10-24 15:54:01 的最高901提交结果：官方Score **.6922、排名第五**；men .7510、doc .8124、micro .6499、macro .5555，第一名 .7389。来源是用户消息，没有再次访问官网；原不可变接收凭据保持，见[独立结果记录](../submissions/round2-best-20261024-user-result.json)。对应原dev .754361398，差.062161398。两关系分项分别差.088880/.101686，约占总落差76.6%；不能将单次差距外推为本地.8必定超过第一名。

用户明确继续固定原dev≥.8，同时允许扩展方法、多分支/worktree、多代理迭代。每个独立模型≤10B，允许集成；按原文献隔离和公共v2比较，禁止将dev标签作为训练资源。新方法的价值以完整验证与互补贡献判断，不以是否原创判断。官方上传仍暂停。

| 新工作线 | 独立分支 / worktree尾目录 | 首要机制与职责 |
|---|---|---|
| span_normalization | explore/span-normalization / explore-span-normalization | 调研并实现跨度检测与HPO规范化联合学习或显式span方法；区别既有BIO/GLiNER简单追加，不只换encoder名称 |
| document_relations | explore/document-relations / explore-document-relations | 调研ATLOP/DocRE等成熟文档级关系建模，联合患者及表型多处提及；区别已失败的局部pair CE/简单max-MIL |
| grounded_generation | explore/grounded-generation / explore-grounded-generation | 调研有原文证据的患者表型生成与独立规范化；明确区别旧8B开放HPO编号失败，优先复用9B及规范化资产 |
| patient_schedule | explore/patient-schedule / explore-patient-schedule | 原标准CE与输入保持，固定fresh四轮训练；完整终点与原两轮及强组合比较 |
| schema_extraction | explore/schema-extraction / explore-schema-extraction | GLiNER2.5原生联合实体/关系抽取，独立HPO规范化，先完整内部控制 |
| qwen3_relations | explore/qwen3-relations / explore-qwen3-relations | 原生Qwen3-8B的strict train32关系SFT及全文F1；不同基座控制，不称首次全注意力 |

每线维护自己的WORKLOG/experiments/NOTES/evidence，先用已有历史排除重复，随后完成可运行初版和训练内完整对照，不停在调研报告。允许复用成熟代码、模型和现有已核资产；源码/许可/版本保存。常规工程修复自主处理，未知状态恢复原作业，避免重复启动。模型/大型raw留外部，关键结果及时commit/push并核远端。

主线程负责共享数据/评测来源、GPU调度、跨线比较和融合。既有全文indexed与FLA完整质量继续收口；患者α.5已完整负：primary .661496611109低原 .663841820660，alone .636714244169，不扩大/不扫α，终审进行中。新路线CPU/来源准备并行，GPU先盘点空卡再按实际成本运行，不抢现有作业。强方法若在固定训练内留出上改善，则验证另一折、训练full64并在原dev作最终验收；新训练输入上限须实际编码核验，不能截样本或套旧cap。

本阶段无新best，原dev仍.754361398；新分支未达标前不合并实现，主分支持续保存统筹和真实进展。

三个新worktree已从main `0c0672d` 创建，对应远端分支均已push/ls-remote核同SHA。span负责人 `/root/span_normalization`、docRE负责人 `/root/document_relations` 已启动；生成负责人 `/root/sequence_recover_0518` 正将旧α负实验收尾后转新worktree。span拟先显式起止联合打分+固定规范化，docRE独立选择完整文档多提及机制；尚无新GPU或新分数。

首轮资源排程：span与docRE复用.3已有Biomed/BioLink/Sap资产，span先获≤600+30窗口（尚未实际启动），释放后docRE接续；两者CPU准备并行。A100 GPU2拟留给患者证据生成，当前仅旧α小补包I/O且连接暂退避，不等同GPU作业。A800新全文80与.100 FLA80原质量运行不受影响。显式分类器保存紧凑全候选分数便于后续学习融合，生成器保留raw/IDs与解析状态；不把大张量/日志放Git。

启动前统一划分：三新路线初版均使用F2+F3 train32→F1完整16篇/39患者，第二折再train48→F0。span/docRE原先拟F0方案尚未实际GPU或数据准备，已在启动前协调修订，方法/预算不变。公共冻结模型可复用，任何监督别名、候选/特征模型仍须核本折来源；不把强train48对照称同训练量对照。三线各自固定非目标对象并报告分项/差值，root在相同F1文献上组装完整融合，避免为对齐重复补训。

GP首个实际训练已09:39:09.637056在.3 GPU0唯一启动proc_03655e70-29ef-4447-9413-84f06e920184/PID1196540，总600+30含丢弃工程probe、fresh8epochs、完整588窗与声明池Sap规范化。train32 1148窗/117170tokens，2266跨度有2263可精确对齐，3条不可表示披露；嵌套仅10，不夸成主要缺口。生成池固定logit≥−2、最终决策>0且不扫阈值，完整声明池分数外存。docRE及生成线继续CPU准备，尚无新质量分数。

GP首入口6.766秒在任何forward前因未列已弃用pooler两权重退出，随后.3短暂离线；原失败完整保存。09:45:05.190878连接恢复后唯一proc_f83b21dd/PID1196954继续同原09:49:09.705568+30截止，未重置预算，数据/方法不变。docRE完整CPU实际1.250秒，train300窗/held156窗；727正对中720有attention支持交叠，7条context=0但两端全提及embedding保留且不删pair，不冒称完整全局注意力。释放后最长doc工程获120+30窗口。生成任务已澄清quote为复制的表型词面、非人工患者证据：87训练451672tokens、监督11884、max输入9210、max监督610；39held max10080。全部身份/targets保留，原生CPU回放120+5及A100 GPU2真实14反向/Adam/三Pareto420+30已授权，尚未新训练。

后备第四控制：在新生成路线的A100显存工程完成前，先CPU准备一次patient-first原标准CE的fresh4epoch/64calls/warmup4对照，独立分支`explore/patient-schedule`。既有[轮数复盘](../research/round2-next-method-gap-20261024/EPOCH-ADDENDUM.md)只发现normal累计四轮曝光负，未发现patient-first fresh统一四轮终点；这不是继续LR已经归零的旧checkpoint，也不是扫一组epoch。固定F2+F3→F1、原124行/输入/schema/CE/LoRA/seed不变，只有统一训练长度与同比warmup/schedule改变；保存最终第4轮，复用旧第2轮对照，不新增中途held挑epoch。CPU/source准备先行，A800是否分配由新生成显存结果决定，尚未授权实际GPU启动。

首轮两条小模型均完整负：GP直接追加 .701293122低强.728697536，GP-only .663575802；新增361概念单元仅51严格TP+6NIL匹配、304未匹配，真正强池外TP4/NIL2。当前只在原强池加GP span置信特征：原a465三个严格nested OOF与101列保留，新增valid/raw-logit两列含低于−2分数，不扩候选/换ID；3新模型与4全量打分277.219秒完成，唯一CPU LGB/F1终点待收。DocRE union .645655645、候选内替换.534242627低N .676546355；候选覆盖469/520，当前选择ATLOP原成熟`*`边界及起始marker表示新对照，源98f9b6a，10:04唯一600+30正式启动，方法与候选其余条件不变。

生成模型A100唯一14反向压力已10:00:29启动proc7d1bc518/66162，420+30；首8微批/1Adam和三族更新已过，最终Pareto峰尚待。独立normalizer624条train-only query16.717秒完整，公共bank仅切片不重编码；LODO117对/11错是训练诊断，不是held分数。后备patient4路线已实际CPU构建强N∪旧P2 .671703123的匹配baseline，低N .676546355；未来两预写端点可择超过强N的最高Score，主并集是否胜匹配2轮另报，不以primary名称否定真正更高单臂。尚未patient4 GPU。

10:12更新：GP严格折外特征F1唯一新fit/v2由.728697536469升至**.730812709670**（+.002115173201），实体mention +.006648、doc +.001813，关系保持；原直接追加失败仍保留。F0实际旧特征102列、强.740572140440，已核旧概率和nested实体逐一复现，现准备相同控制102→104，不照抄F1的101列。

DocRE marker完整8轮64更新178.409秒后仍负：并集.663128103、候选内替换.558854734；较无marker并集+.017472但仍低强N。AUROC .82108→.86579、AP .44811→.54738不能代替最终Score。原两实验95件321669B小包已在.3实际回读逐SHA；后继按成熟ATLOP训练设置研究一个事前固定fresh30轮/240更新对照，其余条件保持，900+30排GP F0之后，只有最终两个完整端点，不挑epoch或扫cut。

生成模型真实14反向/1Adam/三Pareto工程完整通过，峰28.182GB，A100已释放；正式fresh87×2/22calls的3000+30及随后39fresh1200+30、两固定完整质量端点均已授权。原生CPU及模型工程不重跑；10:10:32 A100连接离线发生在任何start intent之前，部署源保留，owner按原10:13:37退避恢复，不冒称正式训练已启动。

患者fresh4轮对照已分配A800并10:08:43唯一启动proc975a3621/PID27842；共享5400秒原TERM11:38:41.825555含四完整持久ACK。唯一watcher143721持续跨轮，owner检查点c0fcff6已推。全部完成后直接沿已授权80fresh及双端点评分，无须再逐阶段确认。

FLA内部F1/F0均有小正后，当前原dev最佳901的四原推断流开展一次完整部署对照：CPU从旧raw重建140578B **byteexact901**，234输入715548tokens/最长5198与原相同；10:10:38唯一proc_e4930517/PID2721779在.100启动，共享1800+30，不在切流时重置。只变实际推断后端及相应硬件/runtime，保留两个原full64适配器、原prompt、解码差异和原患者差集算法；全部234fresh后冻结一个完整预测、一次v2，原best只读。此时尚无新的原dev分数。

10:24更新：GP第二折.740572140440→.739506259147（−.001065881294），mention微升但docF1下降，F1正/F0负，不直接迁原dev。新立固定四折补全：只补F2/F3必要三个模型和评分，原两预测不重做且负折保留参与pooled，保留门为pooled严格升且至少3/4fold非降。当前DocRE30已10:19:15启动900+30，之后接GP补全。新生成正式10:13:49启动，TERM11:03:49.227060。患者4ep第一轮590.267秒完成、520MB完整ACK后原状态启动第二轮；本地watch丢失已独立恢复145075，根因未知/原远端未重训，见H58。

新增成熟方法调研：[GLiNER2.5预训练联合抽取](../research/multitask-relation-readiness-20261024/README.md)，已固定公开模型/代码/许可和ModelScope镜像身份，计划另建schema-extraction工作线。尚无权重加载/GPU/新质量，未知预训练谱系不编造为纯架构比较；现有GPU任务继续原排程。

224完成审查索引已root增量独核并发布，7组包含两工程/五已完质量；新261成员增量包10:23实际从.100恢复全SHA通过，需先恢复既有217备份，再叠新包，原901/官方accepted对象不变。当前活动质量不混进这批。

10:34更新：完整901替换为FLA的原dev结果**.7538954523085053**低原.754361398199，独审b952104c通过负结案；234全fresh/1056.450秒，实体不变、关系417（净2加1删），TP321不变/FP+1，保原901。既有内部完整并集两折曾正，因此另预写唯一完整901∪新fba缓存控制，仅通用关系并集/一次v2，不搜索子流或挑选新增关系；该新研发控制不改前一负结论。

DocRE30两终点仍负：并集.669661581305、候选内替换.631836310231，虽胜各自8轮却低N及历史完整支持组合。AP从.44811→.54738→.65608只说明排序特征潜力。另准一次纯CPU低容量配对meta-OOF：F1四文献组12train/4held，固定完整public-new支持baseline，base-selected vs加DocRElogit/valid，train-fold scaler+LR(C1/.5)，8fit/双pred先共冻/2v2旧0，只有同时胜完整baseline与base-only才扩F0。F1标签作为meta训练明确披露，不称盲测，不扫cut。

GP F0终审a3f855负完成；后继F2/F3四折补全已10:32:46.912唯一启动，三个fresh模型/八评分，原TERM10:47:46.982215。生成首轮11updates/982.831秒已完，原单进程继续第二轮；患者前两轮完整ACK后第三轮已接续，root10:34核独立watch145075仍活。新schema分支3078f54已建立/推送，owner开始原生API/完整坐标/ModelScope资产；.100卡预留，尚无GPU预测。

[上下文规范化查重](../research/context-normalization-gap-20261024/REPORT.md)证明Top8/context CE与严格OOF CE入树均已实际负完成；所谓top3扩池来自公开分类头，不能误称Sap top3。当前不为同名重排再开GPU，先保新机制和正在运行控制，见H59。

10:44结案与接续：FLA原dev整体并集为 **.753601147974**，低原901；两新增关系均FP，独审562c81ef，原全替换负结论保持。两个质量小包共55成员已从.100实际下载、逐SHA落盘及重开核验，回执0c911b12。DocRE低容量meta-OOF为 **.674068552245**，低base-only完整基线.686338981272；三直接控制与meta集中终审954f7259全部负，F1标签用于meta训练明确披露。两新审查与GP F0/FLA全替换暂列224之后待批量收录。

GP补全11步已344.730秒完整，卡实证释放；因.3连接抖动仅断点回收同一91成员ZIP，不重复训练。A100生成已恢复连接确认第二轮17/22步，A800第三轮原监控健康。GLiNER2.5已在.100完成真实strictload/164原生窗合同，最大612词片唯一120+30探针正在运行；若实际成本可容纳，沿预写方案直接完整164窗和固定两质量终点。

10:49决策：GP四折完整pooled **.731169764822**，较原.731545429370下降.000375665；F2微正/F3负，总2/4非降，固定保留门失败。原两折及91件原结果均保留，不因具备可复用GP48分数而迁full64/原dev。

新[Qwen3关系基座查重](../research/qwen3-dense-attention-readiness-20261024/REPORT.md)确认旧train48→F0关系训练和Task1 F1已做，未找到严格train32关系SFT→完整F1；已有R1也为全注意力，不将候选收益归因为attention消融。原生基座8.190735B＋r16为8.234382B；.100既有缓存可复用。实际CPU124train/80held原生token合同通过，训练403151tokens/max5534，held666591/max14322。已建推explore/qwen3-relations自ff676a0，DocRE负责人转线。CPU与标准CE/indexed工程通过后唯一420+30压力；若24GB/估计容纳，直接fresh2ep2400+30、两完整ACK及80fresh1200+30，预写单臂/强fullsupport整体并集双终点，不逐阶段等待确认。OOM则保失败协调A800，不截样本。

10:59新机制结果：GLiNER2.5完整F1两端点均负（N∪new .672407143643、关系单臂.459793526949，实体固定），496原件/完整覆盖/规范化与双冻结终审dac8bde8通过。root随后另立固定 `fullsupport ∪ (DocRE30∩schema)`，不改两源原负或阈值，唯一CPU新v2由 **.686338981272→.690618521538**；新增22=10TP/12FP、micro+.003702、macro+.013417，完整16/39/1189、非关系对象保持，独审ae9c4631。首装配pair-reader schema错误发生在pred/gold前，保原失败并只改reader；见H60与新分支explore/relation-consensus（47e27e4/6bcf4c5）。

新共识只在内部开发F1正，原dev仍901。后继固定F0复验：DocRE32全祖先实际排F0（446e470f），所以不重训，仅原model完整held推断；schema相同公开模型/train32(F2F3)规范化policy亦排F0，F0实际172窗/47患者。先.100短schema再Qwen连续工程/训练，.3补DocRE推断与必要Sap；root只评完整共识F0一次，不为这次复验另扫单臂分数。

长训练已闭环：grounded87×2/22updates实际10:47:48完整，2039.028秒、两完整ACK；39fresh10:54:45启动。patient4四份ACK10:56:59全齐，旧watch正常结束，原owner接续80fresh；不重启旧训练。另有[非思考双seed共识准备](../research/nonthinking-consensus-readiness-20261024/REPORT.md)，只读查重未找到同机制，保留为候选，尚无新推断授权/作业。

11:14固定共识复验已负：真正强F0 .711507580027→.710063635556，新增9=3TP/6FP、micro/macro皆降；同32模型与政策完整预测通过终审3cb75715，按事前F0门关闭，不迁原dev。F1正保留，两份25成员结果包均已从.100实际恢复。patient4主 .662072367414/单臂 .644254142624双负关闭，四ACK/三真实resume/80fresh204.117秒终审8105646e通过；GP四折最终审8829345d亦完成，旧子计划文字残留与正确实际fold数据分开披露。三个新审加入224之后的待批次清单，未冒称已入224。

持续的新机制：Qwen3原probe78.737秒完成真实14FB/Adam，fresh两轮第二轮推进；A800空出后转native双seed共识，明确改用同native G与同A800 runtime而不混FLA，固定160fresh/1800+30于11:10:24启动。primary保强B所有关系，只补双sample一致；另预写M2of3诊断，两完整端点共冻后各一v2，不单评/挑seed。患者response-code层另作CPU准备，保原clinical ID/全文证据，仅增加独立输出别名表与可逆target，不冒称简单无语义ID置换。grounded39在原窗11:13:10完整（39EOS/0cap，1103.215秒），曾预授权的超时尾行恢复未使用；继续原固定归一化/双质量，保留quote坐标供后续独立Task1候选供给判断。

### 11:19 新终点与后继控制

Grounded原39条全部EOS且未触发补跑，254必要Sap查询11.6026秒完整结束；F1固定融合 .681717659986 高于N .676546354663，新关系35TP/63FP，new-only .583496662244，仍低于历史强支持 .686338981。机械元数据接口修复保留原失败，未重跑模型或旧评分；集中终审进行中。主线程另规划将全部接受的精确quote统一加入强Task1基线，尚无该控制质量。

Qwen3正式32更新/两份完整ACK完成，80fresh推断11:14:28启动。患者输出编码CPU准备见[报告](../research/patient-response-code-readiness-20261024/REPORT.md)：原user证据完整，目标可逆映射，监督7170→8138，held最大14570，容量上限14848且不截断。新独立分支预先固定coded-alone主端点、N∪coded诊断端点；A100240+30真实新形状探针通过后可接2ep/两ACK共享3600+30及80fresh1200+30，未把准备报告当作实际训练。

### 11:28 三条关系正候选与固定融合

Grounded强B融合 .695950447742、双采样强B融合 .693653303500、原生Qwen3强B融合 .689312034492均胜同B .686338981272。三条分别独立记录；其中后两条训练/推断完整结束，GPU已释放。主线程在全部输入固定后预写两融合端点，强B保护下至少二源支持达 .701087332488（新增36=18TP18FP，micro .747809152872 /macro .641601241291），严格胜最佳单源；全并集 .684526718071负关闭。见[独立融合记录](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/parallel-relation-fusion/solutions/parallel_relation_fusion/WORKLOG.md)。原dev901尚未变，三条同权重F0准备中，主融合须在F0严格胜B0 .711507580027才保留。

Root另做强Task1的统一quote增量：42新实体11TP31FP，.728697536469→.727242335488，负闭且已审；[记录](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/explore/grounded-task1/solutions/grounded_task1/WORKLOG.md)。小包21成员已实际恢复，融合23成员亦实际恢复；未把任何未达.8实现合并main。
