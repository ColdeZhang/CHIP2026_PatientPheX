# 病例检索工作台账

实验 `E-20261023-008`，独立分支 `round2/case-retrieval`，本轮基点 `00f7da8`。

## 当前状态

- **目标**：原 train64/dev16、冻结公共评测 v2 all/text 的 Score ≥ 0.8；未达标，不合并主线，不上传官方。
- **原dev两种口径**：原独立检索 CPU k21/.35 **.625073738**（首日 .613517597）；共享公开Pheno/CRF/GLiNER实体+Qwen full64关系、固定新候选支持追加 **.736526555**（独审245c1f00通过），高于此前6401组合 **.732257765**。新结果是共享组件组合进展，不冒称纯检索分；未达.8。
- **最新可信内部结果**：抽取LoRA+冻结Sap筛选给CRF+Sap新增18跨度（含HPO12TP/6FP）。原规则关系不变的实体消融 .711354→**.714060535**；同本路线k21/.35完整关联 .657639→**.662396726**。12产物独立逐字节重装、评测/覆盖/格式全部通过，不能把两种关系配置混为同一对照。
- **最新结论**：fresh full64唯一epoch002小幅改善原dev关系：273TP/60FP/180FN vs SFT48 271/64/182，Score +.004549；仍较fullLR331TP损召回、未达.8。64条strictJSON、466CID全合法、0触顶回退，固定输入/decoder/955实体。204严格OOF四事实树仍负，既有结果不重跑。
- **跨折复验**：固定抽取+Sap筛选fold1的规则关系不变实体消融 .644796→.643815；同k21/.35完整链 .619760→.622463。未重现稳定实体收益，不扫阈值/epoch；7臂0错警，具体见证据。
- **当前任务**：fresh full64训练及唯一同A30原dev新臂均完整结束，训练40.77min/2ep/68updates；推断341.386秒（生成314.437秒），A800 GPU2和A30 GPU1均已fresh核空释放。先冻结pred aa107790…再唯一新评分 **.715759605**，独立终审81493ebe通过；[完整结果](evidence/round2/qwen35-full64-dev-result.json)。原训练/推断watcher均结束，勿重启。
- **身份桥接已结案（负）**：同4127 OOF训练/1166外层候选/947实体，fresh BF16 CE baseline **.652284538**→bridge **.641437506**（−.010847），低rules **.711353522**，不迁移原dev/不扫阈值。217TP/133FP→225TP/123FP虽改善micro，macro .561125→.494091使Score下降。两臂4ep/1032真实更新、133.25+133.43s；双pred先freeze、唯一新两臂评分，独立终审b1dc09f5通过（质量e05bbac2/谱系93905390），配置关闭。[结果](evidence/round2/patient-identity-bridge-paired-result.json)。A30 GPU1 20:24:45.882Z fresh空已交root，watcher28859结束，无GPU后继；原FP16失败与BF16数值修复分开保留。
- **补充固定并集也负**：内fold0 rules∪9B SFT .711354→.704652，15新增只有1TP/14FP；实体不变，无门控/阈值策略，关闭且不评原dev。
- **严格CPU校准已结案**：F1 meta-val固定grid选.30，outer旧.5 .653192→.3 .669003，但低rules .711354，关闭且不迁移原dev。实际2506训练行/594正/297列，完整CPU约44秒，无GPU；原模型/基线字节未变，见[结果](evidence/round2/strict-meta-calibration-result.json)。
- **当前新诊断**：full64目标CPU审计完成，265行真实token/mask exact；CID49.53%/标点28.04%但含1908空/非空决策分支，19/265全空，不能据数量直接改loss。在线loss .114268→.060500不等于最终自由生成拟合。唯一12篇训练prompt的epoch2生成67.126s完成：strict12/12、57TP/6FP/9FN、F1 .883721、exact7/12；只说明训练侧已学习，不是Score/泛化、不据15错误追加epoch。独审a9c0ccb1通过，20:40:37Z A800 GPU2已释放，watcher7739结束；[诊断](research/full64-objective-diagnostic.md)。
- **后继只读结论**：旧48完整Adam/RNG仍在但52步日程已lr0，追加必须新协议；两目标候选与已试负控制分开。12份9B实际manifest均thinking=false，模型卡支持接口但adapter能力未证；若后续测试须先验think闭合再strict final。无新训练；[短判断](research/qwen35-successor-readonly-review.md)。
- **thinking诊断已关闭**：同fixed12/epoch002/greedy/1600只开thinking，首条72.048秒/1600真截断/未闭think、无合法final；root授权据此早停，仅1/12，无fit/不解析targets（launcher仅SHA检查）、不重试。独审67b49e51通过，20:51:01Z A800 GPU2空已释放、watcher30920结束；[结果](evidence/round2/qwen35-thinking12-result.json)。
- **SFT融合库存只读**：原四事实用base非SFT，不能外推否定SFT+LR。旧48/64均非inner OOF；6语义方向可复用、2已装配、0合格32SFT。完整3模型粗估约62min训练，最小一meta-fold约21min+推断，但均非实测/授权；[边界与来源](evidence/round2/qwen35-sft-fusion-reuse-review.json)。
- **单meta-fold SFT+LR控制已完成**：新F1-only HGB同LR+上下文 .673525489→加SFT4事实 .683809711（+.010284）；micro+.051593/macro−.010456，仍低rules .711354/单SFT48 .695444。freshSFT32仅F2+F3，2ep32updates/1149.460s；80F1响应282.502s/587CID全有效，双pred先freeze后唯一评分。不迁移原dev、不自动补两折；独立终审0adcce70通过（actual谱系35a792d5/概率与指标重放通过）。[结果](evidence/round2/sft-meta1-result.json)。A800 GPU2 21:28:40.539Z空已释放，watchers89789/6422已结束，无GPU后继。
- **最新训练侧误差诊断**：严格SFT32→F1 16篇，520canonical真关联仅366在候选；209FN=154缺候选+31CID空+24非空漏患者。55候选内FN仅3在top4丢光目标marker，31所有±500窗均无但全有global；这些是结构代理，不证明因果。关系改进不能修154上游缺口；下一建议优先候选召回，或有界teacher-forced决策排序可辨别检查，当前只读无GPU。[诊断](research/sft-training-error-diagnostic.md)。
- **公开新概念关系F1控制已完成**：固定449新concept、旧1189实体/383关联不动，新增193=72TP121FP；v2 .660942003→.669751257（+.008809）。literal baseline310TP/73FP/210FN与canonical诊断311/72/209相差1alt-ID，评测gold未改。62输出/449CID全合法，203.803s，21:43:55.422Z A800 GPU2空释放、watcher9359结束。独审37c03a29通过，先固定outer0复验，不迁移原dev；[结果](evidence/round2/public-new-relations-result.json)。
- **outer0公开候选复验完成**：旧SFT48 .695443948→raw **.694365855**（新增35TP77FP）→固定cut.3 support **.705717148**（新增19TP11FP/+ .010273），仍低rules .711354。46strict/303CID/0触顶回退、146.695s；旧947实体/276关联全保留，双新pred先freeze244dcb4c再唯一v2。2026-10-23T21:51:38.896207+00:00 A800 GPU2 fresh空释放、watcher44595结束，勿重启；独审df6df5e8通过，raw配置关闭、support进入一次原dev固定迁移。support阈值含pooled64校准边界持续披露。[结果](evidence/round2/public-new-relations-outer0-result.json)。
- **F1固定支持过滤**：只保留新增HPO落在固定cut.3文档实体者，原旧输出不变，Score .660942→raw .669751→support **.682128804**；新增65=41TP24FP，比raw少31TP97FP。这是见F1后提出的开发控制，非盲测；不改已冻outer计划，等共同终点。[结果](evidence/round2/public-new-relations-F1-support-result.json)。
- **原dev唯一支持追加完成**：原6401 .732257765→**.736526555**（+.004268790）；新增35=18TP17FP，旧1004entities/333assoc全保留，关系291TP77FP162FN，micro+.014235/macro+.002840。42strict/276CID/0触顶回退，139.727s；fullfreeze36fd0124先于唯一新arm评分，pred e9b30447，raw未评分。2026-10-23T21:57:55.235482+00:00 A800 GPU2 fresh空释放、watcher9688结束，勿重启；独审245c1f00通过，无GPU后继/不上传。[结果](evidence/round2/public-new-relations-dev-result.json)。
- **条件子集分数诊断完成**：固定12CID/26pair/98分支全部有限，79.071s总/52.680s实际forward，22:11:31.682Z在原22:11:42截止内；22:11:39.721Z A800 GPU2 fresh空释放，watcher94304结束。先完整scores/边际freeze再预写统计：旧greedy6TP0FP0FN，新固定.5为4TP2FP2FN/Brier .103977；样本原本无FN，不能检验补召回或外推Score。首attempt在加载元数据序列化时零forward失败，仅修serializer且同绝对预算内完成，不记质量负；无后续GPU。独审5204f184通过（98数值/26边际/指标exact及同预算修复谱系）。[结果](evidence/round2/conditional-subset-likelihood-result.json)。
- **全80CID条件分数已负结案**：同12prompt/158pair/530分支（旧98exact复用），旧greedy34TP7FP4FN/F1 .860759→新.5 24TP21FP14FN/F1 .578313；回收2FN却丢12TP、新18FP/移4旧FP，Brier/AUC均负。新432总333.460s/forward307.520s，22:22:09.253Z完整、22:22:14.936732Z fresh空释放，watcher99868结束；独审ddd3e009通过。全部530先共同freeze再唯一预写统计，不扫cut/温度、不迁移dev。[结果](evidence/round2/conditional-all80cid-result.json)。
- **固定槽判别完整结案**：fresh32/2ep/32calls1222.669s，单模板base .498008→adapter固定.5 .630248；唯一CPU期望F1对照2.526s→.648033，对自家+.017785但仍低同折旧SFT32 .660942。固定.5只回收4旧FN却丢57TP；expected净增23TP23FP，仍整体少旧30TP、多7FP。不扩full64、不迁dev、不扫cut。双80推断122.715s，23:06:29Z A800 GPU2空释放，watch63323/97632均结束；独立终审0d22930b通过。[完整结果](evidence/round2/fixed-slot-binary-result.json)。
- **记录入口**：[关键实验过程](experiments/E-20261023-008.md)、[研究笔记](NOTES.md)、[证据索引](evidence/round2/README.md)。大量产物根目录 `/workspace/patientphex-runs/round2-20261023/case_retrieval/`。

## 已完成与失败结果

- 全文Qwen关系负：同实体 .615159→.569358，过度归属；原响应和格式回退保留，未采用。
- 原90窗抽取严格配对：base→2epoch SFT 的 span-F1 .398276→.539197；这不是Score。6失败窗唯一17子窗恢复后101响应全合法，原90永久冻结。
- 完整HPO链接和未筛选并集负：SFT补CRF实体、固定原规则关系 .679806；同k21/.35 .600940。无效配置冻结，不扫epoch。HPO43/43 JSON合法无触顶，6越候选项明确未解决。
- 固定Sap筛选小幅正：旧LR未保存，使用原train48/源码/18缓存重建，14,480旧候选p完全相同；新池与旧池共享829项因FP16批次有数值微差但阈值决定全同。固定`.65/.92`只保留18新增；见[完整控制](evidence/round2/sft-linked-filtered-control-summary.json)及[筛选来源](evidence/round2/sap-filter-control.md)。
- 真实optimizer续训状态已独立核验：Adam step/moments、scheduler、RNG恢复完整；独立fresh训练数值轨迹等价未证明，不宣称bitwise训练复现。

## 当前任务与恢复

1. **9B原58完整配对已结束**。原adapter作业 `proc_301c8be3-bba6-4c69-9634-bf4a0dbf752b` 与本地watcher93066均正常结束；58/58严格JSON、0触顶、415/415候选合法、0格式错，生成203.10秒。原base .633859674→adapter .695443948，仍低rules；小证据见[配对结果](evidence/round2/qwen35-relation-paired-result.json)，完整raw/预测/SHA在外存 `qwen35-preparation/relation-quality-plan/`。不重启，不挑epoch。
2. **训练已结束**。fresh204行×2ep/52updates，原作业 `proc_c35c5440-2f32-4842-b4aa-8e444173673b` 与monitor89754已结束。实际1855.10秒，allocated37.48GB/reserved78.50GB分开记录，无OOM；epoch002 adapter SHA8e32daee…，两epoch及Adam/scheduler/RNG全实际SHA在外存 `relation-training-preparation/checkpoint-actual-hashes.json`。完整配置、命令与终点见[完成证据](evidence/round2/qwen35-relation-training-complete.json)，不挑epoch1质量，不宣称新的数值resume等价。
3. **A800204与固定两树已完成**。原作业 `proc_0adf6588-bcd4-4b60-a490-a22e0152c958` 正常退出；204strictJSON/0hit，648409实际prefix精确、生成735.192秒。原1477训练CID全合法，验证397合法/18旧格式失败。609374固定两HGB先双freeze再评：无LLM .653192044→4facts .649124234，分别201TP/66FP与188TP/60FP，均弱于rules .711354；[终点证据](evidence/round2/llm-feature-tree-result.json)，外存 `llm-tree-fold0-results/summary.json`。原GPU2于18:38:45.877Z fresh核空已交root，无自动GPU后继。
4. **A30原dev配对已完成并释放**。两臂64完整/466CID全合法/0触顶回退，same955实体。base .646040894→epoch002 .711210414，仍低fullLR .711324992；无schema/阈值修改。实际339.672+337.989秒，全编排748.516秒；双pred freeze `ad68570c…`先于读gold。正确终态入口外存 `dev-relation-transfer-preparation/paired-completion.json` / paired-summary.json；早期paired-plan是被freeze绑定的历史回收快照，不覆盖其SHA。A30于18:36:44.771Z fresh核GPU1空后交root；[终点证据](evidence/round2/qwen35-dev-transfer-result.json)。原独立检索.625074与本组合.711210分开保留。
5. **8B全文控制已结束且释放A30**。同A30两臂58/58、0触顶/删demo，原58 .618120269→全文 .610344254；947实体与所有候选不变。配对墙钟805.38秒；FP的‘不属任何指定患者gold概念’293→323，‘属其他指定患者’53→70，前者不能直接认定真实背景。双raw与双pred先冻结再评测，原解析回退2→7，未应用schema控制。外存 `full-context8b-a30/paired-summary.json` 及[结果](evidence/round2/full-context8b-paired-result.json)，独立终点审查已通过，报告外存validation/full-context8b-terminal-review。
6. **模型/adapter复制均已结束并关源**。A80015文件→A30共19,329,392,091B，2116.35秒；原monitor96986已退出，独立重读15文件全部size/SHA同源，原source进程停止、端口41236拒绝连接。另本轮epoch002两adapter文件173.18MB以独立allowlist复制18.62秒，全SHA与训练检查点一致，自有41237源也已停止/核关闭。没有残留复制/GPU自动等待任务；[持久副本证据](evidence/round2/qwen35-a30-persistent-copy.json)。A30模型实际目录 `case-retrieval/qwen35-preparation/models/Qwen3.5-9B`，新版runtime保持与旧8B隔离。
7. **环境与来源**。A800只读Python `ontology-round2/qwen-venv/bin/python`，私有 `case-retrieval/qwen35-preparation/a800-migration/runtime` 提供TF5.20.2/PEFT0.21.2；模型 `/dev/shm/patientphex-qwen35-1045-c20223623576` 为已全SHA核验15文件的易失缓存，持久源在A100 `476f81cf-09e7-445b-a7f0-d0e12b12e50f:case-retrieval/qwen35-preparation/models/Qwen3.5-9B`。HF c202236…/MS aa2aaf…，LICENSE差异单独处理；A100 GPU等待计划已停止。
8. **预设CPU schema控制已完整负**。e9867a0a源码在两原始响应冻结后共同应用，base3行/11CID/30声称归属投影使Score .633860→.628354；adapter全合法0变化。原raw/原解析所有SHA前后一致。未知CID/HPA不猜修，原解析仍为主结果；外存 `relation-quality-plan/schema-control/summary.json`。事后聚合显示SFT减少无任一目标gold概念FP60→37、其他患者FP42→13，但比rules少29TP；FN153中113不在固定实体概念池。聚合不用于改提示/候选。
9. **已结案**：整患者自由HPO47×2负在 `patient-set-fold0-bf16-v1/evaluation-summary.json`；抽取跨fold1七臂在 `sft-extract-fold1-bf16-v1/fold1-final-report.json`，均独立CPU重放通过。不继续无收益网格，Qwen3-8B旧模型/环境/checkpoint不覆盖。

## 备份与边界

只修改`solutions/case_retrieval/`，共享总账/HANDOFF由root维护。最近远端已push/ls-remote核验 `203fee2`（固定槽完整训练与双臂推断计划检查点），实际probe终点按阶段提交；间歇401已恢复，无凭据变更。本台账允许滞后自身提交；大权重/预测/原始日志不进Git。冻结HPO、训练与验证文献隔离、禁止官方上传的约束持续有效。

- **固定槽后继 CPU 政策已预写**：主 .5 双臂完成后一次患者期望 F1 集合解码，1557 literal 单元一一对应/0 compound与重复；固定 tie/保留准则，30秒、无搜索。当前只保存源码/计划，未产生该臂质量；[计划](evidence/round2/fixed-slot-expected-f1-plan.json)。
- **资源恢复只读准备**：23:00:05Z 重新全字节 SHA 核 .100 旧48/64 epoch002 两 adapter 共346,364,718B，均与原A800一致。逐文件backup locator及反向恢复命令已保存，未上传/运行恢复、未删除；6份旧optimizer仍只在A800。当前训练活跃目录不动，是否释放重复副本由后继实际收益与root决定。[资源记录](evidence/round2/fixed-slot-resource-locators.json)。
- **固定槽正式训练完整终点**：fresh2ep/32 optimizer calls（含首lr0）、1222.669s、峰allocated25.653GB，未触发partial；唯一epoch002 adapter776ceaf1，Adam/RNG完整。23:03:17Z GPU2空、watch63323结束。真实绑定双80推断plan284e16d3已准备，300+30总限，不重训；[终点](evidence/round2/fixed-slot-binary-training-complete.json)。
- **后继仅CPU接口核查**：F0旧58prompt/415CID/1166pair/47patient，208999tokens/max6872，0超7168/0literal冲突；先逐对象重放F1全80输入exact。实际5.704s，无CUDA/新预测/quality。仅准备二特征互补方案的可行性，binary仍32训练而旧F0 JSON为48（含F1合法），不称同训练规模；[证据](evidence/round2/fixed-slot-F0-cpu-inventory.json)。
- **单次F0 binary32前向已完成**：58prompt/1166pair，adapter776ceaf1不重训；native66.246s/forward43.478s，完整scorefreeze29ee8fbd、p6fe3a795，仅原两logit/p不作.5输出。23:15:25Z A800 GPU2空释放，watch44659结束；已交本体做F1-only两特征LR与唯一F0期望F1评价。本方案无F0gold/新质量读取、不迁dev；[终点](evidence/round2/fixed-slot-F0-forward-result.json)。

- **公开候选训练适配仅CPU准备**：旧124字节/编码完全保留，新88提示622concept/1712pairs（119正1593负），合212/4218/713正；literal候选上界594/818→713/818。实际651512tokens/最长5523/0截断，2epoch54calls估31–33min，超过旧1800s预算。A800余362MB，已核旧adapter双副本可后续按授权释放346MB并final-only，现无删除/训练。严格排F1/F0 learned gate，见[方案](research/public-candidate-SFT-training-preparation.md)。
- **唯一9B与rules交集CPU控制**：保留全部nonrelation、literal逐患者交集，0.184s；.695443948→.702672183，225TP36FP154FN，micro/macro均升，仍低rules .711354。来源/政策/完整freeze与唯一评分已由策略协作者提交401fd73，待独审；[结果](evidence/round2/sft-rule-intersection-result.json)。
- **F0二特征下游结案指针**：本体仅F1拟合logit差+旧SFT hard，唯一F0期望F1 .695443948→.696545689，micro不变/macro+.004407；独审0b44186b通过，无自动原dev迁移。此处复用其结果，没有新评分。

- **公开扩充新授权**：root据真实成本批准fresh公共base+r16、212行×2ep/54calls/3warmup、2400s+TERM30，原JSON loss不变；仅final可恢复checkpoint或soft2340 partial，不自动续训/评partial。新数据独审f0c1a70c与实际trainer CPU6.393s通过。只授权清理A800旧48/64两个已复制权重（346362144B，config不删），须fresh核后满足680MB/余150MB才启动。后继唯一F1旧62提示+原support对照，旧80/383不动；[预写计划](evidence/round2/public-SFT-training-run-plan.json)。
- **交集迁移F1失败关闭**：固定oldSFT∩rules后保留原supported65，.682128804→.661533824；删63中丢40TP、去23FP，micro/macro均降，不迁原dev。策略协作者5229695已push，见[结果](evidence/round2/sft-rule-intersection-F1-result.json)。

- **公开扩充JSON正式运行中**：A800 GPU2 `proc_f062bc89-d476-4007-ac42-caa3aa773bb2`，23:29:00Z启动，native硬截止2026-10-24T00:09:00Z+TERM30，watch10348；外存`public-training-candidate-preparation/formal2ep/{job,monitor,run-plan}.json`。plan910fc8/trainer4a6f/entry独审f421，H32先读原job勿重启。23:28:41Z仅移除两个旧48/64冗余weight共346362144B、已全SHA核.100备份；A800free707502080B/预计final余187502080，config与optimizer未删、旧locator仍有效。详见[启动与存储账](evidence/round2/public-SFT-training-start.json)。

- **原JSON value权重CPU+短probe**：原124数据/完整JSON不变，3220/7170监督token按真实offset覆盖list赋2，含734空列表token；1702跨界token仅连带标点，61.98%仅合并权重系数比例。CPU2.810s/独立offset9464698通过。A100GPU2 `proc_fd56898d` 23:55:27Z启动唯一180s+TERM15真实CE/梯度检查，watch56214，硬截止23:58:27Z；无正式训练/质量。现有A800212训练不改，按[恢复入口](evidence/round2/json-value-weighted-probe-start.json)先读原job不重启。

- **value加权probe失败保留**：上述A100短行追加8masked pad后的loss差超预写容差，完整short数值尚未持久化、最长step未执行，不能报PASS或质量负。23:56:53Z fresh空40339MiB交sequence ICL，不自动重试/放宽容差；[失败与恢复证据](evidence/round2/json-value-weighted-probe-failure.json)。当前212训练51/54、1705.637s正常，继续固定终点。

- **公开扩充212训练完整结束**：两epoch/54calls实际1803.222s，唯一final adapter `cb2f3f8e…`/Adam-RNG已真实SHA核；epoch1/2 loss .135688/.059961仅训练记录。2026-10-23T23:59:18.943Z A800GPU2 fresh空，watch10348结束，不重启。仅新62 F1支持追加计划 `cc2bbb40…` 已按actual trainer/data/inputfreeze/policy绑定，源和原decoder不变，待一次600+30执行；[训练终点](evidence/round2/public-SFT-training-complete.json)、[F1计划](evidence/round2/public-SFT-F1-new62-plan.json)。

- **公开候选扩训练负结案**：旧JSON124+公开新88=212、651512tokens，fresh2ep54calls/1803.222s完整。仅同62新提示固定支持对照 **.682128804→.674046941**（−.008082）；351TP97FP169FN→340TP90FP180FN，micro/macro均降。旧1189实体/383关联保留；新47=30TP17FP，较旧新增集合加1FP/删11TP8FP，不保留/不扩F0或原dev。62/449有效0回退，native209.411s、fullfreeze2d4c2d10先唯一质量。00:03:53.514Z A800 fresh空释放，watch3810结束；独立实际训练1a953b82已通过，质量增量审进行中。[结果](evidence/round2/public-SFT-F1-new62-result.json)。

- **公开212统一独审通过**：训练1a953b82+质量完整重装汇总 `e8627dfe`；唯一.674046941负结果正式关闭，不重评。
- **value权重attempt2实现通过**：两shape内logits exact、CE差≤1.2e−7，全LoRA梯度1.62–1.69%差在预写2%内，非bitwise；同冻结logits poison padding loss/grad exact。原生跨shapeCE也差−.018571，旧失败保留，不归因具体内核。最长5523加权真更新12.181s/peak23.859GB，总96.492s，00:05:03Z A100GPU2空交ontology，watch53301结束。新增独审d71c6e52通过，旧失败ffe5e4c0原样保留；仅工程检查无质量/正式训练；[结果](evidence/round2/json-value-weighted-attempt2-result.json)。

- **value权重唯一正式对照准备**：原124/402594tokens/7170监督不变、值token权重2，其余1，fresh32文献×2ep/32calls/2warmup。A100实际CPU9.571s核所有编码/权重/末4行分母与meta参数，CUDAfalse；2100+30/软2040可恢复partial、只final推断。后继唯一原80 F1对旧.660942003，600+30且不混public62；[冻结计划](evidence/round2/json-value-weighted-formal-plan.json)。源码/计划推送后核UUID空卡启动，探针状态不复用。
