# 第二轮总工作台账与方案索引

更新时间：2026-10-24 UTC（本轮始于10月23日）。主线程负责协调、共同验证、资源、备份和验收；各方案负责人保存完整实验过程，本文只保留关键状态和入口。

## 目标与固定验收

用户要求已完成的首日五方案全部合并推送并清理，再从官方前三的本体规则、病例检索、CRF 分别继续优化。三条路线独立分支/worktree，目标各自在原 dev16 的公共近似 Score ≥ 0.8，独立复核通过后合并 main。**最高901的B100预测已07:54:13确认接收，单次授权已完成；现在继续本地冲0.8，后续上传暂停，用户自行看官方成绩。** 不要求原创，可参考成熟方法。

固定原 train64/dev16、HPO 2026-06-23、evaluator v2 默认 all/text。训练派生资源不能含dev标签；不删样本、改指标或用oracle冒充达标。train64 内额外四折用于选择和解释稳定性，不另设每折0.8门槛。详情：[验证协议](round2-validation.md)、[训练内划分](../data/round2-train64-inner4.json)。

## 三条工作线

| 方案 / 实验 | 最新记录入口 | 首日 dev16 | 第二轮固定 dev16 进展 | 验收状态 |
|---|---|---:|---|---|
| 本体规则 E-20261023-007 | [分支台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/ontology-rules/solutions/ontology_rules/WORKLOG.md) | .612597 | 二层校准 .646244；父100+固定Sap .646360，仅微增 | 独审通过；默认稳定base仍 .646244，未达0.8 |
| 病例检索 E-20261023-008 | [分支台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/case-retrieval/solutions/case_retrieval/WORKLOG.md) | .613518 | 原检索 .625074；共享三public LightGBM实体/新版支持gate＋旧SFT正逆序＋独立患者增量 .754361 | 原检索及新SFT质量复评通过，未达0.8 |
| CRF E-20261023-009 | [分支台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/WORKLOG.md) | .684734 | CRF+语义跨度+LR .700636；GLiNER48 .711325，固定nested .712147 | 48/nested独审通过；患者full64独立对a179微增至.746261；与最新LGB支持gate合成.754361已独审，未达0.8 |

当前实现分别在 `round2/ontology-rules`、`round2/case-retrieval`、`round2/sequence-crf`。各目录内的 `WORKLOG.md` → `experiments/`、`NOTES.md`、`evidence/round2/` 是完整记录入口。**未达标实现不会合并main，所以当前最新过程应看上表的实验分支链接**；主分支对应方案文档仍是启动基点。正负结果均在各分支及时commit/push。

## 当前结果与决定

**02:45参数规则纠正**：官方综述明确每个独立模型≤10B，允许多个合规模型集成。此前按整链总和限制而得出的“必须去BioBERT、不能保留多适配器、Med只能替9B”推论撤销；原真实计数和质量保留。[准确原文/独审](round2-model-parameter-rule.json)，防复发见H47。当时NCR尚未fit，主控制改为保留强B后加NCR；现已完整验证并因跨折不稳关闭，详见第四阶段。

原固定dev最高 **.754361398199**，预测9016863c、统一独审76ada874通过：1061实体/416关系/36患者，men/doc/micro/macro **.792801/.828678/.738780/.657186**。先用新LGB文档概念支持替换public-new旧gate得3049/.754216969（35→54、12TP净增），再仅加患者模型相对a179的6条关系（3TP3FP），总Score再增.000144429、macro略降。两条被新gate删除的旧public关系未恢复。前序3049的46件补包已06:22:27实际服务器恢复；901的36件包已06:42实际恢复，患者adapter第二副本已06:31持久。新授权B100官方单份推断见[提交准备](../submissions/round2-best-20261024.md)。

前序实体模型迁移的.748583439566保留：在原a179固定关系上，将公开PubMed/Bioformer/BioBERT实体的103列HGB替换为固定LightGBM4.6.0，保留cut.3及proper-nested。四内部折先通过既定选择条件，再唯一full64训练/回载/原dev迁移，完整16篇预测先freeze后唯一v2。预测`0e2c26f8…`、实际模型/vectorizer`803b6467…`，统一终审`a1b2c622…`；mention/doc/micro/macro为 **.792801/.828678/.725118/.647736**。相对a179 `.746081965`，6加17删、1072→1061实体，全部391关联及其他非entity字节不变，总分增加`.002501475`。原HGB`42e94115…`及a179预测保持，公共新关系支持仍沿原来源，不因替换Task1擅自删旧依赖。

当前关系为旧Qwen full64正序/逆序、经新LGB支持的新概念关系，再加独立患者full64模型的精确增量；321TP/95FP/132FN。实际新增第二个full64 LoRA，两者共享同一基座，不能再称单adapter重复推断。原患者独立终点a179∪patient及其.746261仍保留；新901是在独立端点完成后另立的唯一明确CPU合成。

固定逆序并集先在F1/F0取得正增量，再唯一迁移原dev；旧dev推断A30、新逆序A800，环境差异已披露，不把全部增量单独归因顺序。原dev历来参与开发比较，不能称盲测；不重算公开新概念支持池。仍未到0.8，第二轮实现未合并；仅最高901按授权完成一次新官方上传。前序baac（.738494）、d531（.737586）、e9（.736527）、6401（.732258）预测和实际依赖原留。当前接收后本地优化见[第七阶段](round2-20261024-stage7-local-optimization.md)，完整训练迁移与全文监督准备见[第六阶段决策](round2-20261024-stage6-decisions.md)，双折互补前序见[第五阶段](round2-20261024-stage5-decisions.md)，前序关系控制见[第四阶段决策](round2-20261024-stage4-decisions.md)；此前方法、分项和负结果见[第三阶段决策](round2-20261023-stage3-decisions.md)；更早见[前期](round2-20261023-early-decisions.md)与[中期](round2-20261023-stage2-decisions.md)。


## 当前控制与后续

官方B全部计算、统一格式与来源审查已完成，唯一上传已获平台接收，见[提交记录](../submissions/round2-best-20261024.md)。三条路线的当前控制、事前条件和优先级只维护在[第七阶段](round2-20261024-stage7-local-optimization.md)，各owner维护真实作业与完整结果。

## 资源与恢复入口

工作树 `/workspace/patientphex-worktrees/round2-{ontology-rules,case-retrieval,sequence-crf}`；产物 `/workspace/patientphex-runs/round2-20261023/`；数据 `/workspace/patientphex-data/`。实际命令、完整UUID、job和日志位置由方案台账/manifest维护。恢复先查原managed作业及GPU进程，不能因会话中断而重启训练（H32）。

| 服务器 / GPU | 下一用途 | 已确认状态与边界 |
|---|---|---|
| 10.5.2.100 A30 GPU1 | 检索全文候选逆序 | R1完整质量已负关闭；新逆序最长通过，后续完整80按实际作业恢复，启动前重查 |
| 10.5.4.7 A800 GPU2 | indexed全文SFT原93f50运行 | 08:07:31启动8400+30，08:22已epoch1/step5；检查点按原SHM→.100持久协议 |
| 10.5.3.12 A100 GPU2 | 公开H5原431edce8正式控制 | 08:21:27启动600+30；实际释放后给患者混合loss工程probe，MCP离线不等于模型失败 |
| 10.5.2.3 A30 GPU0 | 当前无新分配 | Sap/GLiNER B已完成；实际使用前再核空闲与准确缓存，不重复传大模型 |




前序pipeline的端到端资源与薄推断入口缺口见[CRF资源清单](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/experiments/best-pipeline-resource-inventory.md)：Task1 HGB不含GLiNER，Task2仍用CRF/Sap64+GLiNER48候选及Qwen full64 adapter，两池不能无验证合并。旧HGB薄入口与新LGB缓存纯加载入口均已独审；新入口实际输出逐字等于0e2c，仅复用固定候选pool/纯input/旧关系template，尚不包括从原文生成全部候选。

Pheno缓存输入只含允许字段，原16逐文献完全一致；修复了新增校验器对长度保持引号规范化的错误假设，首次失败保留，模型/候选/decoder未改。CPU准备32.88s，实际GPU缓存183.453s；旧投影218.83s仅估计。SFT旧1131s仅token比例估计，实际训练1149.460s，原1800s预算内完成。

已核Qwen3.5-9B公开15文件/19.329GB分别存在A100、.100及A800自有缓存；A800副本在 `/dev/shm/patientphex-qwen35-1045-c20223623576`，不作唯一持久副本，23:28:41仅清理A800旧48/64两个adapter权重重复副本共346,362,144字节，.100目标已再次完整SHA/可读核验；configs、6个optimizer与其他checkpoint原留。释放后实余707,502,080字节、预计final后余187,502,080；存储ledger及逆向恢复locator `5053b0c9…`在检索产物中，locator已准备但未执行实际反向恢复，不冒称双份仍在。A100正式实体训练使用独立路径，预算内保留epoch1/2及可恢复partial。其他模型/依赖遵循自有目录、共享环境只读、目标卡启动前重查；不为下载偏好重复已校验传输。

## 核验、合并与备份

[独立核验索引](round2-verification-index.json)当前206项、4755唯一引用，SHA`78b0c884…`，07:24发布外部live索引；本次同步Git中原191副本至同一已审字节。新增15组425引用，旧191对象/4330引用保持。独审`8793abf5…`见[索引状态](round2-index-integrity.json)；当前最高901/.754361已纳入，新官方B及其后继质量仍待各自终点。

索引1,440,385字节；3.53MB新源码/小证据快照留仓库外，不收权重、完整预测、赛事原数据或大日志。患者新完整训练/原dev质量、SFT32全文后继与pure-load补包未作为本批终点。此后纯加载缓存入口已独审通过，由总台账另记，不回填05:20候选冻结时点。191归档已05:42实际服务器恢复全部4349成员；旧182已验证备份链保持。

五条首日历史已真实merge到main，solution目录与原尖端一致；14项公共测试、4项检索测试通过，旧五分支/worktree已清理。证据见[合并清单](day1-20261023-merge.json)。本轮三实验分支持续commit/push，只有达到固定0.8并独立核验的实现才合并main；共享规则、索引和进展正常进入main。

[Git bundle](round2-git-backup.json)：main `dc0003e`、本体 `011d409`、检索 `44636bd`、CRF `490ba68`，9,879,161字节，SHA `bbcc90b1…`。01:55:07已从服务器实际下载、bare clone、核全部refs及fsck，外部 `git-restore-check-0148/verification.json`。这是打包时点的已提交状态；后续提交继续GitHub核验，bundle不含未提交文件/后续运行/大产物。

[审查材料备份](round2-review-backup.json)：206索引、4755支持引用及14辅助记录，共4770成员，包10,375,410字节/SHA`4ffe0614…`；清单1,154,594字节/SHA`00a018b3…`。07:46:38实际远端下载并逐成员大小/SHA通过，收据`a1ca3ea2…`，同时核必含当前索引与独审。固定词表/配置和短诊断沿旧证据保留，不含完整预测、训练输入标签、大raw或权重；旧191/182恢复链单独保留。

[已恢复的前序.748583产物备份](round2-best-dev-backup-0e2c26f8.json)：预测`0e2c26f8…`、实际full64 LightGBM/vectorizer`803b6467…`、完整源码/计划/指标/终审、固定4.6.0 wheel及旧a179依赖链，共58成员、3,827,082字节，SHA`288622b9…`。05:05:17已从服务器实际下载并逐成员核SHA，外部`dev-artifact-restore-check-0503/verification.json`；完整manifest16,801字节、SHA`168b0c39…`。Git仅存compact定位与真实恢复结果，不复制成员目录。包不含大概率表、训练输入、NN权重，不冒称独立可重训环境。

[纯加载复现补包](round2-best-dev-reproduction.json)：固定缓存入口3.463秒逐字重建0e2c，独审53a91d2f通过；新增候选pool/HPO及源码/合同/审查16件、2,908,699字节，SHA6aa98c35。05:27:32已实际服务器恢复全部成员，外部`pure-load-restore-check-0526/verification.json`。模型/template/wheel沿58件包、纯input沿旧a17972件包，不重复制；仍未闭环原文候选生成。

旧[.746082恢复包](round2-best-dev-backup-a179a5de.json)保留a179预测、原HGB模型`42e94115…`、提示/响应与旧关系依赖链，共72成员/520,986字节、SHA`d9af976a…`；00:43:26已实际服务器恢复。不能因实体模型替换而删掉仍被公共新关系支持使用的历史模型/来源。

旧[.738494恢复包](round2-best-dev-backup-baac8186.json)含86成员/317,919字节、SHA `ce29ef50…`，00:13:01已从服务器实际恢复；作为不变实体模型、原候选及旧关系链的追溯入口保留。

旧[.737586恢复包](round2-best-dev-backup-d531f2ee.json)（23成员/74,339字节、SHA `c9ee2d34…`）、[.736527恢复包](round2-best-dev-backup-e9b30447.json)（46成员/301,892字节、SHA `32fcfcf6…`）以及[.732258恢复包](round2-best-dev-backup-6401a24e.json)（37成员/1,593,942字节、模型 `db2fc288…`）均已实际恢复。旧包沿链保留不变，用于未改变的关系与历史资源追溯，不把旧HGB当作新三来源模型；[更早.728795恢复包](round2-best-dev-backup-16cd8352.json)亦保留。

主线程下一步：继续第七阶段本地优化到原dev0.8，核各路线完整终点与备份；单次官方上传已完成，不重复上传或查询成绩。当前最高仍低目标，第二轮实现未合并。206索引/4770成员已实际恢复，新B与接收凭据单独增量保存。

## 本次唯一官方提交已完成

最高901方法的B预测50ef5daa于07:54:13 UTC获平台确认，额度5→4；完整100/244、0格式错误，统一终审cd7a6040。原始[接收凭据](../submissions/round2-best-20261024.json)与[方法、恢复说明](../submissions/round2-best-20261024.md)保留精确版本及平台文件名。没有查询官方分数，用户自行查看。后继优化暂停已解除，其余上传继续暂停。
