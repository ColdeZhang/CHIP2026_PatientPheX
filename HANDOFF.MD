# 接手必读：已确认的坑与避免方法

最后核对：2026-10-23。**每次执行新任务先读本文，再看 [WORKLOG.md](WORKLOG.md)。** 本文保存长期经验；当前任务进度只在台账维护。旧工作已按用户要求整体归档，新项目从干净的 `main` 开始。H01–H08 是归档旧实现的经验，不是当前代码的待修清单；H09 的环境配置已修复。

## 快速定位

- 任务：全文表型位置识别、HPO 标准化、指定患者表型归属。
- 约束：模型不超过 10B；HPO v2026-06-23 的 HP:0000118 分支；禁止外部人工标注表型数据集。详见[比赛资料](docs/competition/README.md)。
- 首日五路线已全部合并；当前三路线本地优化见 [第二轮总记录](docs/campaigns/round2-20261023.md)。旧 CPU 复现及全部旧产物仅保存在 [归档](archive/legacy-20261023/ARCHIVE.md)，不作为新实验默认起点。
- 完整证据：[接手审查](archive/legacy-20261023/docs/REPOSITORY_REVIEW.md)、[审计统计与脚本](archive/legacy-20261023/docs/review/2026-10-23/README.md)。原 README 的历史路线结论不能直接作事实依据。

下方 H01–H08 中的 `src/`、`data/` 等路径均相对 `archive/legacy-20261023/`；链接已指向归档证据。

## H01：同名分数不代表同一评测口径

`src/evaluate.py` 忽略 HPO ID，docF1 定义不同；`zip` 不校验文献 ID/长度，漏尾部文档可得满分；空患者、NO、复合实体处理也有差异。

**避免方法**：先校验文献配对、患者覆盖和指标定义，再比较分数；明确训练/验证/诊断/官方标签。近似诊断也不能标成官方成绩。证据：[评分反例](archive/legacy-20261023/docs/review/2026-10-23/evaluation_results.json)。状态：未修复。

## H02：验证标签可能通过划分和词典进入训练

关系 dev 的 79 篇文献全部与 train 重叠；增强词典新增的 2,035 个词条全部出现在训练金标中。全训练回测无法证明泛化；`--self` 也不会创建验证划分。

**避免方法**：先按 `pmc_id` 分组划分，再用训练折构建派生资源，保留 split manifest。词条重合只证明重合，不证明作者全部构建过程。证据：[关系样本](archive/legacy-20261023/docs/review/2026-10-23/relation_samples.json)、[词典重合](archive/legacy-20261023/docs/review/2026-10-23/dictionary_overlap.json)。状态：未修复。

## H03：本体文件名正确，版本和概念范围仍可能错误

仓库 `data/hp.obo` 为 2026-09-01，比赛要求 2026-06-23；旧输出还存在分支外、废弃和畸形 ID。

**避免方法**：记录 HPO 版本和校验值，验证分支、alt_id、废弃项及 ID 格式；不能用较新本体的检查冒充指定版本的判定。证据：[本地核对](archive/legacy-20261023/docs/competition/LOCAL_NOTES.md)、[输出审计](archive/legacy-20261023/docs/review/2026-10-23/output_audit.json)。状态：未修复。

## H04：总行数和 JSON 可解析不能证明提交完整

坐标是原始全文全局位置，不能直接无间隔拼接段落重新编号；患者 ID 只在文档内唯一。`local_pro_v10.jsonl` 虽有 100 篇，却漏 14 位指定患者、多 1 位未给定患者。

**避免方法**：按文献 ID 配对，用 passage offset 检查原文切片；用 `(pmc_id, patient_id)` 检查指定患者覆盖。区分“无表型”与“遗漏患者记录”。证据：[输出审计](archive/legacy-20261023/docs/review/2026-10-23/output_audit.json)。状态：异常旧产物保留作证据，未修复。

## H05：词典补丁可能无效，或把误报移到别处

七个 v10 新增双字母缩写被后续长度限制全部过滤；ASD 硬编码覆盖成错误 HPO；长词内部和嵌套短跨度产生误报。直接放开双字母、删除黑名单或固定最长匹配也会损失正确结果。

**避免方法**：修改前跑具体反例和最小消融，同时检查精度/召回及 ID，之后在独立划分验证。证据：[缩写审计](archive/legacy-20261023/docs/review/2026-10-23/abbreviations.json)、[规则消融](archive/legacy-20261023/docs/review/2026-10-23/rule_ablations.json)。状态：未修复。

## H06：同段共现不是患者归属，否定也有作用域

同段所有表型分给所有患者会混人；只依赖给定 mention 会漏跨段信息；固定前 30 字符的 `no/negative` 判断可能误伤后句。无 HPO 的患者表型关联还需按官方规则保存原文。

**避免方法**：分别验证实体发现与患者归属，保留跨句/跨段证据；区分患者、家属、一般讨论、否定和无 ID 表型。证据：[规则归因](archive/legacy-20261023/docs/review/2026-10-23/rule_audit.json)。状态：未修复。

## H07：模型实现失败不能证明技术路线失败

关系样本标签与候选 HPO 不一致；NER 只用起点 F1 选模，末尾特殊 token 可把实体解码为 `(5,0)`；新增实体未标准化，同起点不同长度实体也会取错。

**避免方法**：先验证标签、解码、标准化和端到端指标，再判断模型价值。一次分数下降不足以证明 NER/关系学习不可行。证据：[模型诊断](archive/legacy-20261023/docs/review/2026-10-23/model_diagnostics.json)、[完整审查](archive/legacy-20261023/docs/REPOSITORY_REVIEW.md)。状态：未修复。

## H08：能找到输出，不等于能重建实验

旧机器绝对路径、`rel2/rel` 不一致、权重和词典构建链缺失；历史“0.86→0.32”“0.394”“达标”等缺提交凭据。CPU 输出只保证患者表型集合一致，列表顺序可能不同。

**避免方法**：记录实际代码/配置/数据版本、命令、环境和产物位置；集合型结果按语义比较。重要证据不能只放 `/tmp`；本地文件、commit 和已核验远端备份是三种不同状态。按 [AGENTS 第 8 节](AGENTS.md#8-及时提交与远端备份)及时提交、推送并核验，最近检查点见台账；被忽略的大产物另行保存。证据：[复现记录](archive/legacy-20261023/docs/review/2026-10-23/output_audit.json)、[审计 manifest](archive/legacy-20261023/docs/review/2026-10-23/manifest.json)。状态：旧训练链未修复；审计关键统计与脚本已归档。

## H09：推送成功后，单分支 fetch 配置仍可能阻止 upstream 解析

本环境最初只配置了 `+refs/heads/main:refs/remotes/origin/main`。`git push -u origin work` 已成功创建远端分支并写入跟踪配置，但本地没有可解析的 `origin/work`，查询 `@{upstream}` 仍失败；这不代表远端提交丢失。

**避免方法**：检查 `git config --get-all remote.origin.fetch`，必要时用 `git remote set-branches --add origin work` 增加当前分支，再 `git fetch origin work`；其他分支按实际名称替换，保留原跟踪范围。验证 upstream 能解析、ahead/behind 符合预期，并用 `git ls-remote --heads origin refs/heads/work` 核对远端提交。不能仅凭 `push -u` 成功就认定跟踪配置完整。

**证据/状态**：2026-10-23 当前环境已修复；`@{upstream}` 为 `origin/work`，ahead/behind 为 `0/0`，本地与远端均为 `cb682f8`。该修复属于本地 Git 配置，不随文档提交传播；新环境需重新检查。

## H10：云账号登录、天池登录和比赛提交权限是三个状态

短信验证后，阿里云页面还可能需要确认已绑定账号；云账号页面跳转成功不代表比赛可提交。首次通过手机号入口登录时，天池 `login=true`，但比赛 `signed=false`、`showSubmit=false`。按用户要求退出并用 `ALI_ACCOUNT/ALI_PASSWD` 中的用户名账号登录后，比赛改为 `signed=true`、`showSubmit=true`。不能把手机号入口选中的账号当作用户指定的用户名账号，也不能将前者的报名状态外推到后者。

**避免方法**：使用用户明确指定的账号，分别核验天池会话和当前比赛/赛季权限，再决定是否提交。切换时退出阿里云 SSO 并移除旧快照；新会话保存到仓库外并验证恢复。证据/状态：2026-10-23 10:53:14 UTC，用户名账号独立 context 恢复后仍为已登录、已报名且可提交；未上传预测。浏览器进程可能跨轮次退出，需要从 profile/快照恢复；每次提交前复查，不能承诺永久登录。恢复方法见 [会话说明](docs/competition/SESSION.md)，最新状态见台账。

## H11：认证请求报错可能带出请求头，直连也不等于浏览器网络路径

本环境 Playwright `APIRequestContext.get` 访问天池时出现 IPv6 `ENETUNREACH`，其默认异常详情会包含请求头；已登录浏览器页面内的 `fetch` 则可成功读取同一接口。

**避免方法**：优先使用当前天池页面的会话请求，设置超时并只返回白名单状态；捕获异常后仅输出异常类型，不打印完整堆栈、请求头、Cookie 或会话响应。证据/状态：2026-10-23 已用页面内请求完成登录和比赛权限核验；原始网络错误已绕过，认证内容未写入仓库。

## H12：可用 MCP 服务可能通过工作区客户端提供

原生工具列表未暴露用户所说的 `start_skill`；实际工作区有 `mcp_client.py`，通过环境变量 `MCP_URL/MCP_TOKEN` 可发现并调用 GPU 服务器工具。没有找到技能正文，不能声称读过或按技能执行。

**避免方法**：先查原生工具与技能，再按用户线索查工作区客户端。当前无凭据副本为 [scripts/mcp_client.py](scripts/mcp_client.py)；只输出工具结构与资源状态，禁止打印认证参数。使用空闲卡前核对 UUID 和进程，仅使用自有目录和环境。

## H13：远端模型下载与 MCP 大文件传输有实际限制

远端 Hugging Face 与镜像连接失败，本机可下载；MCP `write_file` 较大请求返回 HTTP 413。直接把模型或数据一次写入不可行。

**避免方法**：按用户最新偏好，先复用已核验副本，新下载优先 ModelScope（魔搭）；固定对应revision并核对权重、配置与tokenizer的SHA，不能只凭模型同名认定等价。当前Qwen3.5-9B已从魔搭下载完成，15文件共19,329,392,091字节逐文件实际SHA通过，源端四个权重分片的大小/LFS SHA与指定HF版本一致；许可证文件差异单独记录。魔搭缺少所需版本或无法核对时再尝试原始HF来源。2026-10-23另一条已验证路径：本机对官方HF resolve作HEAD取得临时官方CDN地址，远端有限并发Range下载；BioLinkBERT-large 1,334,073,393字节在256秒完成，全SHA与官方一致。临时签名URL不能写入Git或长期manifest，CDN ETag不能代替模型SHA。直连不可用时，再本机下载校验，按不超过512 KiB原始字节MCP分片传输、合并核验；优先续传，不重复下载。模型和全量日志不入Git。远端默认Python/依赖不一定存在，使用已验证解释器和自有环境，不修改他人环境。具体来源/版本见[研究索引](docs/campaigns/round2-research.md)及方案模型manifest。

远端部署需等待文件传输真正完成再启动依赖作业：Sap新特征准备曾在异步部署完成前启动，因脚本尚不存在退出，未载模型、未用GPU。启动器现要求成功部署的plan/source收据及前阶段完成标记，失败保存在外部 `ontology_rules/public-sap-feature-control/attempt1-launch-race/`；不能以“已发起传输”替代依赖就绪，也不能把这次工程失败记成模型质量结果。

## H14：官方附件元数据中的相对路径不等于可下载 URL

将 `/file/race/documents/...` 直接拼在网站根路径下会下载到 HTML，解压出现 `BadZipFile`。

**避免方法**：使用已登录官方页面实际下载按钮，保存浏览器下载对象，核对 ZIP 内容和 SHA。数据来源和指定 HPO 的同字节核验见 [数据说明](docs/data/README.md)。

## H15：选择上传文件不等于平台已接收

选择文件后先出现预签名成功响应，文件名可见，但额度尚未改变；本次稍后才出现历史记录且额度从 5 变为 4。不能据最初响应重试，否则可能重复消耗额度。

**避免方法**：上传前保存文件 SHA 与尝试凭据；单次选择文件后检查对应历史记录及额度变化，再记录平台接收。未确认时先查历史，不盲目重传。首日提交阶段仅核实接收；后续按用户指令单独回收官方成绩。证据见 [首日提交清单](docs/submissions/day1-20261023.json)。

## H16：复合 HPO 关联在金标与预测中必须用一致评价单元

dev 金标包含 6 项 `HP:ddddddd;HP:ddddddd` 关联。公共评测初版保留整串，而规范预测拆分为原子 ID，造成评价不一致；部分模型的关系训练同样漏拆金标，产生错误负例。

**避免方法**：仅拆分完整合法 HPO 复合串，不能盲目按分号拆自然文本或 MESH。公共评测已在 `92fa1f8` 修复为 `patientphex-public-approx-v2`，14 项测试通过；旧指标保留，新结果另存。训练侧未修正的路线在实验报告明确限制。alt ID、范围外金标与正文口径仍属官方未知项，不能悄改。证据见 [评测复核](docs/experiments/day1-evaluator-review.md)。

## H17：候选构造的训练自包含会掩盖模糊检索价值

模糊路线初版训练候选仍能使用当前训练文献自己的金标别名，即使特征统计排除当前文献，也使正例几乎都是精确候选。按文献排除候选别名后，模糊输出增加，但本次总体得分下降。

**避免方法**：同时核查候选生成和特征构造的折内隔离；分别记录精确/模糊的产出与正确率。该现象不是 dev 泄漏，也不能据一次下降否定模糊路线；首日选择较好的 v1，保留 v2 负结果和复现入口。证据见 E-20261023-003。

## H18：GitHub 认证失败有时短暂发生，仍须核验实际备份

本轮推送曾返回 Invalid username or token，API 也返回 401；按用户提醒间隔重试后恢复，后续 push 与 `ls-remote` 核验一致。不能仅凭一次失败认定必须更换令牌，也不能把本地 commit 当作远端已保存。

**避免方法**：先保存本地检查点，有限次间隔重试；不输出凭据、不强推。如仍失败，明确标记未推送并使用已授权的服务器 Git bundle 备份。恢复后统一推送并核对远端 SHA。当前备份工具及实际恢复验证见 [提交与备份](docs/submissions/README.md)。

## H19：小验证集的最佳路线不保证官方最佳

首日 dev16 上 CRF .684734 最高，但同批 B 官方最高为本体规则 .603338，CRF .590650；排序差异主要体现在患者关系。两篇多患者 dev 文献包含12/36患者，事后分组诊断显示其对路线排序影响很大。线上线下同时改变训练规模、预测样本和评测器，不能仅据差值认定泄漏或某个因果解释。

**避免方法**：将官方结果精确绑定文件/版本；比较统一评测版本（首次本体回执的本地分仍为 v1，不能混入 v2 比较）。下一轮使用按文献分组的多折和病例分组对照，保留原验证结果，不因看到官方分数就删样本重选验证集。证据见[首日官方分析](docs/campaigns/day1-20261023-official-analysis.md)。

## H20：提交页完成状态与成绩页分数是不同证据

提交页仅显示文件名、时间和“测评完成”；完整 precision/recall/F1 在“我的成绩”接口，UI只显示四位小数。成绩记录通过 `fileId` 对应上传列表的 `id`，不能只按页面顺序关联路线。

**避免方法**：读取官方完整精度，核对比赛/赛季、fileId、文件名、时间、大小和冻结 SHA，保存白名单字段，不保留账号或认证数据。CDP重新连接后的视口可能恢复为窄屏，桌面“我的成绩”菜单消失；在同一连接设置视口再导航，或使用已确认的 `/score` 路由。接口与字段见[会话说明](docs/competition/SESSION.md)。

## H21：同一文献集合不代表训练字节及随机过程一致

第二轮本体首次远端 fold0 重建保留原 train64 文献顺序，后续批次使用冻结 fold 文件的排序；文献成员相同且仍与验证隔离，但负例随机抽样顺序改变，导致同配置 Score .702788 与 .703333 不同。这不是四舍五入，也不是评测口径变化。

**避免方法**：远端直接传输冻结输入，重建时同时检查文件 SHA，而不仅比较 pmc_id 集合。种子不能消除输入顺序变化。记录实际运行源码/配置/数据字节，保留第一批结果，不能在原结果上改写来源。证据/状态：本体分支 `0673f85` 记录两批差异；最终四折与原dev均按冻结输入 SHA 核验，第一批只作历史选型。详见[本体最新台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/ontology-rules/solutions/ontology_rules/WORKLOG.md)。

## H22：公开的比赛自训权重可能已包含本地验证文献

Agriv公开模型卡声明自训标签来自官方80篇，其中包含本仓库固定dev16；下载可用和参数合规不等于可作本地留出验证。其五折模型的折成员也尚未核对，不能按同名fold0推定与本仓库划分相同。

**避免方法**：区分公开通用预训练与本赛已微调权重，先核对训练文献来源/折成员，再决定用途。本轮只读取公开模型卡和元数据；本地比较需从允许的预训练基座按自己的train48/train64重训。没有使用该比赛成品权重。来源和固定revision见[研究索引](docs/campaigns/round2-research.md)。

## H23：按文件名记录源码哈希会覆盖同名依赖

CRF GPU入口曾以 `Path(path).name` 为字典键，CRF与语义编码器都叫 `run.py`，后写入者覆盖前者，manifest遗漏了一个实际依赖。模型输出仍存在，但仅凭该manifest不足以精确复现。

**避免方法**：用稳定角色名或仓库相对路径作为键，保留全部依赖的SHA；旧记录另附补充证据，不能静默改写。CRF分支 `9d3da67` 已补实际远端依赖哈希、角色键实现及同名文件回归检查，详见[方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/WORKLOG.md)。

## H24：LLM的JSON可解析，不代表患者键与输入一致

两条Qwen路线遇到含冒号的患者ID：提示中的“ID: 别名”被模型整体用作键，例如 `OII:3: II:3`，真实ID为 `OII:3`。这属于身份字段格式失败，不能误报为患者人数翻倍或模型没有作答；也不能把未匹配结果静默丢弃。

**避免方法**：用可逆P001等局部编号或明确结构化身份字段，严格验证患者/候选集合与截断状态。本体22窗口中3个失败窗口仅按格式、未看gold选择一次修复，最终22/22通过；旧raw、回退版本、重跑来源及SHA分别保留，修复后分数仍下降。证据见[本体分支实验](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/ontology-rules/solutions/ontology_rules/experiments/E-20261023-007.md)。

## H25：外部原论文的临床事实未必属于比赛标注范围

原论文表格通常比给定输入的caption包含更多临床信息，但直接补患者关系并不必然加分。按训练侧预先冻结的严格患者匹配/表型词典政策，在固定内fold0给rules新增31条、给LR新增29条关系，全部为该评测口径下的FP，Score分别 .711354→.698422、.642034→.631899；没有改Task1实体。

**避免方法**：先确认标注范围，再评估外部证据的收益，不把“原论文确实记载”当作gold包含的保证。训练诊断中99.185%的gold关系可在给定正实体中找到，当前优先改进原文跨度和患者归属。保留表格来源及负结果，不把单一政策/单折负结果泛化为所有表格无用。证据见[CRF表格对照](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/evidence/round2/tables-fold0.json)。

## H26：WordPiece标签合法，不代表解码跨度落在词边界

神经NER曾输出 `lipomatosis` 中的 `lip`、`microcephaly` 中的 `haly` 等词内碎片；token标签、字符切片和JSON格式都可以通过，仍是错误表型边界。独立审计已核实训练标签往返覆盖99.22%，没有大规模mask/窗口故障，不能把这些碎片误判为整个训练链路损坏。

**避免方法**：单独检查推理跨度的词边界；保留训练标注确有词内跨度的例外，不一概修改gold。仅按train48冻结的过滤控制让三种神经NER在内fold0改善约 .004–.0075 Score，仍未超过CRF；已交方案负责人，尚未宣称跨折稳定或全局替换。证据见[CRF实验记录](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/experiments/E-20261023-009.md)，完整审查在仓库外 `validation/ner-audit/`。

## H27：短代词的子串定位会生成合法但错误的患者指代

全文共指首轮训练输入pilot的15窗都能解析JSON，但 `he` 的普通子串出现次序会数到 `the`、`mother`、`phenotype` 内部。输出原文切片存在，仍不代表正确患者指代；还出现把表型短语当患者称谓、把已指定历史病例混为当前病例的身份错误。因此没有扩大到验证折。

**避免方法**：对短指代检查完整词边界；用唯一匹配的原文上下文定位，拒绝无法唯一定位或患者身份不清的项。新增共指仅在已验证坐标起作用，不能把 `they` 等新词面自动广播到全文。先做训练input-only身份检查，再冻结验证协议；不依据gold修指代。修正原型同3篇复验保留4处新增称谓，逐项身份检查通过；另35项被严格定位规则拒绝，召回很保守，尚未证明验证集得分收益。首轮响应SHA `19f5cd1ad1670017e79b11642124dc457aed8a04b3afdef317a609260727156a`；源码快照和取舍见[CRF方案实验](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/experiments/E-20261023-009.md)。

## H28：长期worktree不会自动收到主分支的新规则

三条实验分支从同一基点创建后，主线程持续更新共享规则与踩坑记录，但分支里的根文档仍停在早期版本。直接在该worktree读 `HANDOFF.MD` 会漏掉主分支已确认的新坑，方案自己的实验记录却可能是最新的。

**避免方法**：新任务同时检查当前main的共享文档，可用 `git show main:HANDOFF.MD`；方案源码/实验记录仍读自己的工作分支。阶段检查点干净后正常合并main的共享更新，检查是否意外改变方案源码或评测器并推送。三路线已安排同步，不要求每个总台账提交都触发merge，也不把未提交实验混入同步。规则见[AGENTS第1节](AGENTS.md#1-开始任务与恢复上下文)。

## H29：合法HPO编号和原文quote，仍可能互相不对应

整患者自由生成实验同时输出HPO ID与原文quote，格式/分支/坐标检查均可通过，但仍出现 `short stature` 配成 `HP:0001250`（Seizure）、`cardiac abnormalities` 配成 `HP:0000252`（Microcephaly）。不读验证gold，仅按指定HPO及train48唯一词面核对：微调模型42个原文支持项中14项编码冲突、28项词典未知；后者不能据此判错。另44条合法响应中26条为空，说明格式改善也不等于召回改善。

**避免方法**：分别核验原文定位、HPO语义与患者归属，不能把任一项通过代替其他项。优先让模型选择带名称/定义的有限候选或可逆编号；自由生成编码若要保留，应单独验证其标准化质量。词典匹配仍不证明患者归属，歧义词面和未匹配长quote不能强行修正。本次仅保存诊断与负结果，未依据gold打补丁或追加重链接评分；证据见[检索实验](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/case-retrieval/solutions/case_retrieval/experiments/E-20261023-008.md)及外部 `patient-set-fold0-bf16-v1/diagnostics/exact-quote-id-summary.json`。

## H30：第三方训练loss能正常运行，仍可能因位置参数错位反向优化

实际A800私有环境的 `gliner==0.2.22`（`modeling/base.py` SHA `b3ef985596410c07a447f18f5e199339b97ca4f603441a9ff36f9242d949e5bd`）中，`SpanModel.loss` 调用 `_loss` 漏传 `prob_margin`，使 `negatives=1` 落到 `label_smoothing=1`，所有标签变成0.5。CPU最小复验：两有效logit `[2,-2]`、标签 `[1,0]` 加一个padding，库loss为1.126928且正例梯度为正；声明的有效跨度BCE为0.126928且正例梯度为负。正常退出和有限loss不能证明监督正确。

**避免方法**：监督训练前核对实际版本的调用签名，以可手算正负样本验证loss、梯度方向及padding mask。已绕过：方案 `45e107c` 改用自有入口显式计算logits上的BCE和有效跨度mask，不改共享库；实际A800 CPU独审已通过正负梯度、极端padding、masked标签翻转不变及按总有效span归一，修复报告SHA `0f5032956fde4e7f1fb75e30062951db2bfea247386dbd8141a1a6209a6daac2`。本次在GPU单步及正式训练前发现，既有零样本推理不经过loss，因此不受影响。结论只适用于该已核对源码，不泛化到所有GLiNER版本；数值证据在外部 `validation/gliner-supervised-review/actual-runtime-probe.json`（SHA `de238b865de637eea05dccaafb6458d0be0df1bdb9bed5cb15a3934a7fc9e1d1`），方案入口见[CRF实验记录](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/experiments/E-20261023-009.md)。

## H31：BERT风格标记名不保证存在于实际tokenizer词表

准备新的关系MIL时发现，当前BiomedBERT词表没有 `[unused1]`、`[unused2]`、`[unused5]`、`[unused6]`，直接查token ID均得到UNK。不能从模型属于BERT系列，就假设这些标记有独立embedding；仅在字符串中插入标记，也不能证明编码后保留了预期身份。

**避免方法**：先检查实际词表与完整tokenization；需独立标记时显式注册special tokens、固定随机种子扩展embedding，并在最终编码中核对跨度、标记及截断。保存tokenizer并回读完整词表/special map，不能只保存扩展后的模型权重。当前MIL入口 `b47cd62` 已添加4个标记（ID30522–30525，词表长度30526）、使用 `mean_resizing=False`，3949个出现跨度及原窗内患者标记均保留；保存/回读、loss梯度和纯删除装配独审通过。证据：外部 `validation/relation-mil-review/preparation-review.json`，SHA `7e1a6acc5402a23a51c263c3b2098ed896c9534b9353817b0caffe64fdfc49d6`，源码与CPU记录见[本体方案](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/ontology-rules/solutions/ontology_rules/evidence/round2/relation-mil-run-plan.json)。这是训练前接口修复，尚非质量结果；旧患者CE中的标记字符串可被拆成普通WordPiece，因此不能据此宣称旧实验无效。

同类实例：A100的GLiNER CPU检查曾写死 `[ENT]`/`[SEP]`，但模型配置实际使用 `<<ENT>>`/`<<SEP>>`，在断言阶段失败。检查脚本改为读取真实配置后通过，未改模型或依赖、未使用权重/GPU。首个失败与修复保存在外部 `research/a100-resource-readiness/attempt1-marker-assertion/`；最终CPU报告 `3faa8885…`。不能把测试脚本的错误假设当模型不兼容。

## H32：工作区重连和代理中断，不等于远端训练已经停止

2026-10-23 18:49工作区短暂重启后，文件与提交仍在，但协作注册表只剩一个被中断的代理；A30上的GLiNER正式训练进程仍连续运行，恢复读取时已完成两轮。仅看本地代理状态而重新启动，会重复占用显卡并可能覆盖产物。19:52:39再次短暂出现environment unavailable，恢复后主分支d9f5d5b与工作树仍在；BioMed原进程继续到第6轮，三个方案都核实没有重复启动。

**避免方法**：恢复后先核本地Git状态、方案台账与已保存job/manifest，再读取原managed进程、实际checkpoint和目标GPU的UUID/进程。确认已有作业状态后才恢复监控或重建代理；不要以代理消失推断训练失败，也不要重新跑已完成实验。若确需重启，使用独立输出目录并记录旧进程终态。当前原作业 `proc_b554a946-3034-416e-94b7-bc12084e6b44` 已确认持续运行，未重启；恢复入口见[序列方案台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/WORKLOG.md)（检查点 `31361f9`）和[总记录](docs/campaigns/round2-20261023.md)。

同类传输故障：MedGemma首次私网复制时，本地MCP初始化503触发controller的`finally`，过早停止仍供下载的源HTTP进程，随后fetch报连接拒绝。**监控失败或UNKNOWN只表示观察失败，不能自动清理健康远端作业。** 保存已知job和绝对截止；仅在fetch确认终态、明确停止或截止时清理源，两端各自受绝对TTL约束。修复后3项CPU边界检查及增量独审通过（`validation/medgemma-runtime-readiness-review/transfer-recovery-source-checks.json`，SHA `558ccde4…`）；唯一断点恢复722.052秒内完成19件8.64GB，03:14:31取得完整ACK。10件继承此前完整核验/未变文件证明，9件本轮SHA，不能冒称19件均重新hash；原失败和600秒未启动旧恢复计划保留。完整copy终审 `4b9e48c0…`，恢复不重下公共权重、不代表GPU或质量通过。

## H33：标准库辅助函数也依赖实际 Python 版本

GLiNER-BioMed 的新启动器使用了 Python 3.11 才引入的 `hashlib.file_digest`，而实际系统解释器为 3.10.12，首次在校验文件时退出，尚未加载模型或使用GPU。模型库能导入不代表外围启动器的标准库调用兼容。

**避免方法**：按运行命令所用解释器核对版本，对新增辅助函数先做相同环境的CPU检查；跨3.10/3.11的文件哈希使用分块 `hashlib.sha256().update()`。已修复兼容调用，wrapper SHA `2688f27b…`；实际3.10.12哈希检查通过，重启作业也通过文件绑定并进入推断。专门CPU自检发生在重启后，不能记成启动前已执行。首次失败、原源码和修复记录保存在外部 `sequence_crf/gliner-biomed-nested/attempt1-pythonhash/`、`attempt1-failure.json`、`actual-wrapper-hash-probe.json`；模型/解码协议未改变，不能将启动器失败记成模型质量负结果。后续质量结论见[序列台账](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/WORKLOG.md)。

同类实例（2026-10-24）：ICL adapter传输源使用默认Python3.8，HTTP handler调用3.9才支持的 `str.removeprefix`，源端请求即退出，目标端表现为 `RemoteDisconnected`。不能只检查目标机或把该症状直接归因网络。改为兼容的 `startswith`/切片后，实际两文件18.432秒完成并核完整SHA，源服务已停止；失败保留于外部 `research/retrieved-json-example/formal/attempt2-python38-transfer/`，成功凭据 `formal/adapter-transfer.json` SHA `fd5ed20b4b3658333e231db3d5d0e1b41b8faf52273a58cf2b04621e5bf28644`。辅助源服务器同样要按实际解释器检查，不只检查模型进程。

## H34：文件哈希不能恢复当时未提交的可变文档

生成式NER研究读取方案台账和实验记录时，这两份文件含尚未提交的变更。报告保存了当时SHA，但文件随后继续编辑，记录的Git HEAD也不含这些字节；本次检查现有提交与快照后无法精确恢复。主体研究脚本、报告及已有实验结果仍可核验，不应把两份背景版本缺失夸大为全部结果失效，也不能宣称完整来源字节可复原。

**避免方法**：需被长期引用的实际源码/记录，在读取时保存小型不可变快照或先提交；运行manifest同时记实际SHA与可恢复位置。跨代理审查遇正在修改的文件，应复制所读版本，不只记录分支HEAD和哈希。现有报告/manifest原样保留，两项例外列于外部 `validation/index-next-20261023-2010/declared-dependency-verification.json`，86项索引中明确标注；不能用后来版本冒充原件。

## H35：FP16梯度暂时溢出需要与真正的数值故障区分

患者身份CE首次正式运行在第一轮触发新增的“梯度必须有限”断言，原失败日志没有记录batch序号。实际同顺序复验在第10个batch重现：loss为有限的0.825509，2个梯度tensor溢出；原生GradScaler会跳过该次更新、将scale从65536降至32768，Adam步数和权重保持不变。新增断言在scaler处理前终止，改变了旧实现的恢复语义。最长batch单步通过也不能覆盖后续batch的数值路径。

**避免方法**：保留AMP的skip/backoff语义并正确协调scheduler，或统一采用硬件支持的BF16策略；同时检查有效loss、真实参数更新和数值状态，不能只检查退出码。此次两对照臂统一改为BF16、参数和Adam仍FP32；实际正式顺序前三batch及最长batch均有限，非零学习率下参数更新，诊断状态丢弃后重新fresh训练。旧FP16失败/来源独立保留，不记作质量负结果。证据：外部 `case_retrieval/qwen35-preparation/patient-identity-bridge-cpu-preparation/bf16-repair/actual-diagnostic-complete.json`，SHA `171e5f82356d3f84e75797fd7666df7b1d9dac987cc4261444ce47cf861e35c7`；正式新终点以方案台账为准。

## H36：第三方候选生成器会因跨句可变状态静默漏项

PhenoTagger v1.2 的 `build_ngram_testset_filted` 把参数 `Ngram=8` 在遇短句时原地减小，后续长句继续沿用变小的上限。它正常返回且坐标合法，因此仅检查退出码和格式无法发现召回损失。固定16篇无标签输入的CPU对照中，原70,971候选，改为每句独立 `min(8, sentence_length)` 后75,790候选，增加4,819项、影响72段；52,185个token的原文坐标均通过，修复后的坐标集合包含全部原候选。

**避免方法**：复用候选代码时检查跨样本参数是否被修改，以前置短句/后置长句的构造例和输入级集合对照验证。只修局部上限，不借机改过滤或阈值；事前固定修复版并保存上游差异，不必为了原样复现再跑已知错误版质量。上述只是无标签候选检查，尚不证明新增候选正确或得分改善。证据：外部 `research/public-phenotagger-reuse/native-pipeline/candidate-census.json`，SHA `aac89791d14b7984361ee692940927772f67d8e1f4a8ce86a1d1e4255c203f0e`；固定函数 `1569e545…`，后续质量状态见方案台账。

## H37：放宽上游阈值，不保证经过重叠消解后的候选集合更大

公开PhenoTagger同一批64篇缓存中，将原生NN门槛从>.95放宽到>.5，raw接受6124→10047，但最终canonical仅保留6061个旧键、新增2157个、丢掉3个旧键。原native的重叠分组和缩写处理会受新增候选影响，不能把“更低阈值”当作最终集合的严格超集。

**避免方法**：分别核对raw接受数、最终集合的交集/新增/丢失；若实验目标要求保留旧候选，明确冻结旧新集合的并集并保存真实来源，而非默认为保留。此次在质量评测前固定旧.95∪新.5的8221键，同键flags按真实OR、分数按观测max，保留旧独有3键。后续重新训练选择器仍可能删除或替换实体，不能将该实验解释成只做实体加法，也不能仅凭分数变化归因某一步。无标签计数与来源核验见外部 `ontology_rules/public-phenotagger-lower-gate/incremental-review.json`，摘要SHA `e3c570fc62c5d6e5ce08ab6841bb7e73beaebdb55cd9474189f837821417f51b`；质量结论另见方案台账。

## H38：分别分词再拼接，会改变条件似然的真实边界

在固定Qwen3.5 tokenizer的12条提示、98个合法患者子集续写上，分别 `encode(prefix)` 与 `encode(value)` 再拼接，有95个分支不等于完整字符串分词。JSON引号、括号及分隔符可以和前一段字符合成token；文本看似相同，不代表模型条件前缀和被评分token相同。此问题在CPU准备阶段发现，未将错误分词用于模型质量结论。

**避免方法**：先对每条完整合法续写分词，再取得候选间真正共同的token前缀，按causal next-token位置计算后缀概率，并固定分支结束分隔符。检查所有分支的完整token复原与分歧位置，不从字符边界推断token边界；也不能把只枚举固定JSON顺序的子集归一值称为无条件患者概率。实际98分支的共同前缀/后缀契约已通过，源码 `9b2a33b3…`，证据为外部 `case_retrieval/qwen35-preparation/sft-meta1-fusion/conditional-subset-likelihood-cpu/token-contract.json`（SHA `e1d170ab…`）；实际GPU数值、耗时和质量另按后续终点记录，本条不宣称其已通过。

## H39：模型加载成功后，记录元数据也可能使任务退出

Transformers的实际loading info可包含Python `set`。早期Qwen入口已为此实现显式JSON序列化，但新的条件似然入口重新用了普通writer，加载模型后、任何forward之前再次因 `set is not JSON serializable` 退出。该次没有分数，不能记成模型数值或质量失败；新入口复用已验证的writer后，在原绝对截止内完成98分支。

**避免方法**：公共运行辅助逻辑沿用已核接口，新增manifest写入用真实返回对象或包含空/非空集合的契约检查。对支持的类型显式转换，对未知类型报错；不要一概 `str()` 隐藏数据结构。保存失败源码/无forward证据，核实原进程和GPU已释放后，用独立attempt目录恢复，不覆盖失败产物或重启重复作业。既有helper检查在外部 `case_retrieval/qwen35-preparation/loading-info-serialization-fix.json`；本次仅序列化修复提交 `0a2fada`，模型/98分支输入不变，终点与实际耗时见检索方案记录。

## H40：逐文OOF资源不自动满足新验证划分的隔离

准备train32（F2+F3）→F1的新实体判别时，旧四fold HGB给F2/F3的概率、旧outer0内层候选都对各自目标文献隔离，但其训练资源均包含新F1的16篇。直接拿它们筛选或构造新训练样本，会让本应留出的F1信息进入下游；“是OOF”不能替代完整资源依赖核查。

**避免方法**：验证集改变或增加下游模型时，沿候选生成、词典、门控、特征模型的训练文献集合核查新留出集；不仅核对被评分文献自身。此次在使用前已拦住，改用F3→F2、F2→F3及train32→F1资源，两条训练链与F1交集均0；仅依赖公开固定权重的无标签缓存可共享。新HGB只用train32拟合，其对自身训练样本的筛选明确标为resubstitution，不称OOF或泛化召回。旧资源对原目标仍有效，不据这次复用风险否定旧结果。证据：外部 `sequence_crf/entity-slot-feasibility/isolation-check.json`，SHA `f794767a48ce41e547736dc387401f816ed8588883626e7c9cc98b2e284eb59b`；只核文献ID/资源来源，未进行新质量评测，详见[序列分支记录](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/WORKLOG.md)。

## H41：复用加载代码时，也要复用已验证的后端环境配置

私有 TensorFlow 2.12 / Transformers 4.30.1 环境中，`import tensorflow` 成功不保证 Transformers 的后端探测认为 TensorFlow 可用。BioBERT 新启动器遗漏旧加载入口的 `FORCE_TF_AVAILABLE=1`，5.375秒后在 `TFAutoModel.from_config` 退出，尚未构建编码器、加载H5或forward；这是此前已遇到的环境问题再次出现，不能记成模型质量失败。

**避免方法**：复用入口时检查完整命令和环境，而非只复制模型代码；在首次导入 Transformers 前设置已验证的后端开关，不在已缓存后端状态的进程里补设。CPU检查继续隐藏CUDA、共享依赖只读。此次仅在独立attempt2补上上述变量，原模型/配置/词表和检查代码不变，20.054秒完成201变量、121,587,826参数严格恢复和4条合成短语接口检查；两次共25.430秒，原失败证据保留。结果 `research/public-phenotagger-biobert/attempt2/native-h5-result.json` SHA `545eadcb6f9d5419aece46ab89031d5dcc27b347f30a2949c22c5e75d47aee5f`，独审 `validation/public-biobert-native-review/review.json` SHA `f954be78a92fae7a88c720ea153aaa2edb6d8cb8e7c64f4860cc8a9baef3806d`；仅证明实际加载/接口通过，未证明赛事质量。

## H42：验证 loss mask 时，不能混入骨干模型跨形状的数值变化

JSON 值加权的首次探针比较591 token与追加8个masked pad后的599 token，因loss差超过预设容差退出；当时未保存同形状原生参考，不能判定加权实现错误。独立第二次诊断发现，原生full CE跨形状本身变化−.0185713；在各自固定形状内，selected/full logits完全相同、weight2 loss相同，LoRA梯度相对L2差1.62–1.69%，通过事前2%容差，但并非逐位相同。骨干变化的具体原因仍未确认，不能直接归因BF16或宣布任意padding等价。

**避免方法**：分别验证三个问题：同一输入形状下的新旧loss实现；固定logits上被忽略位置的loss/梯度屏蔽；骨干跨形状的行为。冻结logits并污染8个忽略位置时，loss/梯度均完全不变；最长5523 token的真实无padding更新通过，head保持冻结、三类LoRA参数实际更新。保留第一次失败，数值诊断状态丢弃后再fresh正式训练，不把探针通过当质量改善。证据 `validation/json-value-weighted-probe-review/attempt2-review.json` SHA `d71c6e5298a6a3a5a94e57a8a15a2806e1e1cd725ba15de859f6dc96ecc6675a`；实际96.492秒、GPU已释放，后续训练状态见检索方案台账。

## H43：缓存检查器必须适配实际运行版本的数据结构

FLA 数值探针将实际 Transformers 5.20.2 的缓存 `dict[int, Tensor | None]` 当作 Tensor 列表遍历，首个原生块前后向后对整数键调用 `detach()` 而退出，尚未进行 fused 数值比较。这是检查器故障，不能记成模型或 kernel 质量失败。

**避免方法**：按实际版本的容器结构取值，保留层号、键、shape 和 dtype；未知类型、漏键或意外空状态应明确失败，不能简单跳过非 Tensor 项而假通过。修复先在 CPU 用实际容器和混合 dict/list/tuple/Tensor/None 检查，再在原绝对预算剩余时间内恢复独立 attempt。此次只替换缓存采样与结构比较，原 probe、计划、输入和数值门槛未改；后续单块合同通过，全模型另因原始 logits 超容差停止，两个原因分列。统一终审 `validation/fla-kernel-terminal-review/review.json` SHA `b78eec326865b71ac2c41e96b3ae8b7f2423928d3139e95716ae0a4f948ba2c1`，源码与完整结果见检索分支 `solutions/case_retrieval/fla_kernel_readiness/`。

同类tokenizer实例：Transformers 5.20的`apply_chat_template(tokenize=True)`实际默认返回BatchEncoding；R1准备脚本将其当list，4.791秒在CPU断言退出。实际`input_ids`与显式encode原本完全相同，明确`return_dict=False`后，80行原生token契约5.582秒通过。原失败保留，模型/模板/输入不变；接口容器假设不能被写成模型不兼容。证据为`research/deepseek-r1-0528-qwen3-8b-feasibility/cpu-preparation/`与统一终审`8fa2f637…`。

## H44：检查点通配符会误收同前缀的备份回执

顺序增强第一轮完成并持久导出后，controller将回执命名为`epoch-001-export-ack.json`。旧trainer用`glob('epoch-*')`判断最新检查点，该普通文件被误认为最新epoch，导致第二轮在模型加载前退出；第一轮权重、优化器和RNG完整有效。这是恢复入口故障，不是训练数值或质量失败。

**避免方法**：模型目录与回执使用不同命名空间；发现冲突时，先保存原失败与回执字节，再将回执移至不匹配的`export-ack-epoch-001.json`，核对所有同前缀项确实都是完整检查点。此次仅修改controller两处回执basename，trainer、数据、epoch1全状态与原绝对截止保持；3项实际CPU反例和持久ACK核验通过，增量独审 `validation/candidate-order-warmstart-entry-review/ack-filename-recovery-addendum.json` SHA `baa8be3b795510c40e8d3b73ff898d81f979dc3079bace42d884080fbab01a11`。后续恢复必须按台账的新controller入口，不能误启保留的旧失败入口；修复通过不代表第二轮训练或质量已完成。

## H45：完整公开权重也可能存在旧序列化布局差异

NeuralCR官方fastText文件完整大小1,345,465,351字节、SHA `e9947f25…`核验通过，但其首词`the\0`位于84字节、prune字段位于88字节，与固定fastText 0.9.1的保存/加载布局相反。不能把下载完整当作可直接安全加载，也不能在未知结构上试着交换字节。

**避免方法**：先以无模型解析器核完整666,817词条、两矩阵shape、实际参数和EOF，再对照指定版本源码。仅在本文件精确前提满足时，独立派生副本交换84–96区间内的4/8字节，保留原件；前84字节、96后全部字节、词/count/type顺序及两完整数值矩阵均核对不变。派生SHA `fa7509ec…`后完成唯一原生CPU加载及四合成接口，实际10ensemble/60变量数值精确相等；H5加载的`np.array_equal`不应冒称另做了内存字节SHA。统一终审 `validation/neuralcr-native-terminal-review/review.json` SHA `050b512b8c91cfa655e4edf6d386342e5f7aab71600812718dae04cab08d1508`。这是该固定文件的兼容处理，不是通用修复，也不代表赛事质量改善。

## H46：部署参数库存还包括预处理和非神经学习状态

旧组合清单只加神经权重，遗漏了PhenoTagger实际调用的NLTK POS模型271,206权重、CRF/类型LR、树选择器及经验表。替换Task1选择器也不自动移除旧97列HGB：公共新关系支持仍依赖它。该补账不改变预测，却会改变参数余量的解释。

**避免方法**：沿实际推断调用链清点共享/替换/备用组件，分别列独立学习权重、树结构/经验表、固定知识和派生表示缓存。神经合计明确写小计；不能将缓存每个坐标重复算成独立学习参数，也不能把未拟合新头预先当作已实测。此次实际结构检查1.063秒，无predict/fit/gold；完整库存 `research/nonneural-deployment-inventory/inventory.json` SHA `f9e2718eda4d0d6f12fb8201378f883ad5873ff3b189adfbed7f818aa7a18dde`。原报告按全链10B计算的“剩余32,482,590”只是被H47撤销的过严预算推论，不能再用于禁止集成；实际组件计数与来源仍有效，库存应按每个模型披露。

## H47：核对参数上限的作用对象，不能自行改成流水线总和

此前将网页“模型参数不超过10B”解释为整条流水线所有模型之和，进一步要求NCR替换BioBERT、禁止与旧9B合用MedGemma，并限制额外适配器。这些限制比已缓存的官方明确文字更严，错误地缩小了可研究空间。

**避免方法**：先查精确作用对象及已公开的获准用法。官方综述PDF第4页§3.1写明“Each individual model was limited to 10 billion parameters”，第10页§4.2介绍Qwen3-8B与DeepSeek-R1-0528-Qwen3-8B共同集成。以**每个独立模型≤10B、所有资源披露**执行，MoE单模型总量和基座/适配器/学习头仍须真实核计；流水线总和是资源信息，不是该门槛。PDF SHA `e583c8b0…`与原文定位见[纠正清单](docs/campaigns/round2-model-parameter-rule.json)。02:45发现时去B实验尚未fit，已保留其静态准备并改为保留BioBERT后测试NCR增量；已有质量终点及运行中的患者优先训练均不改写。

## H48：新版AutoTokenizer可能按类名重建并改变发布的处理链

R1-0528-Qwen3-8B发布配置标注`LlamaTokenizerFast`，但当前Transformers5.20.2实际构造`LlamaTokenizer`，重建BPE、normalizer、Metaspace预切分与SentencePiece风格decoder，未保留发布`tokenizer.json`的ByteLevel链。两个AutoTokenizer接口给出同样IDs曾被当成准备通过；后续生成解码出现字面`Ġ/Ċ`才发现，实际80/80输入IDs都与发布原生文件不同，Auto往返解码0/80，发布原生/generic各自80/80。词表和special IDs相同并不能保证分词相同。

**避免方法**：GPU前同时核发布文件与实际运行对象的完整处理链，并用独立原生加载器比较真实输入编码、解码及特殊token边界；不能只比较同一错误入口的两个API。不假设所有模型都逐字符无损，预期规范化必须与发布处理链一致。此次已定位，使用发布JSON的generic修正核验中；raw/generic各自往返正确不自动等于两者IDs相同，新入口须直接核全部80行及序列化缺省字段。旧1024/4096作业、原输入与输出保持，限定为TF5重建tokenizer下的工程记录，不能当发布模型的格式/质量负结论；已有2644 IDs正确解码后仍只有2/8候选键，原strict合同不通过，不能补空列表冒充模型答案。实际CPU11.833秒，无权重/GPU/质量；统一纠正审查`validation/deepseek-r1-format-cost-review/successor-and-interface-correction-review.json`，SHA`4cc5b614…`。

后续修正CPU7.932秒通过：80行raw/generic/chat-template IDs直接全等、往返及special一致，按顺序mergepairs与原发布文件相同；新first2014/max5689，原长度2171/6002不得沿用。证据`published-interface-v1/cpu/actual-result.json` SHA`d0aace8e…`，新GPU/质量另记。另一次同环境17.716秒核Med与Qwen3.5各80纯输入，其Auto/native编码及双decode均相同；不据R1故障否定旧模型。该结果仅覆盖实际文本：Qwen序列化组件/未用音频special存在差异，不能扩大成所有输入或序列化字节等价。

## H49：先解析后保存会丢失最需要诊断的失败响应

患者优先F0首次固定58推断在第24条`3563609:relations:8`遇到患者矩阵键检查失败。runner在`parse_final`后才写raw，因此只保存了23条成功响应；失败条的原始文本和token IDs丢失，无法确认缺了哪个患者、是否有未知键或其他具体内容。调用顺序只支持此前的EOS/非截断/JSON门槛已过，不能冒称已逐项读取失败响应。训练两轮和完整状态仍有效，这不是质量负分数。

**避免方法**：每条生成结束即原子保存原始IDs、解码文本、实际EOS/cap/超时、输入身份及耗时，再做严格解析；校验错误写独立状态，不覆盖raw。此次原rc1/130.785秒、23条缓存和trace保留；后继使用独立目录、复用23条并重新生成其余35条，失败行的新输出不等于找回旧响应。有限缺键弃答政策在任何F0质量前另行冻结，原strict失败不改写，不补负答案或评分不完整子集。修复与完整质量当前待实际验证；恢复入口见[序列方案F0记录](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/round2/sequence-crf/solutions/sequence_crf/experiments/patient-first-F0.md)。

## 更新方式

新问题使用稳定编号，写清触发条件、影响、避免方法及证据。修复时更新该条状态并追加验证版本，不删除防复发经验。待核实猜测不得混入“已确认”；大量案例移入对应实验或审计记录，只留链接。
