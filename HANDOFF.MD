# 接手必读：已确认的坑与避免方法

最后核对：2026-10-23。**每次执行新任务先读本文，再看 [WORKLOG.md](WORKLOG.md)。** 本文保存长期经验；当前任务进度只在台账维护。旧工作已按用户要求整体归档，新项目从干净的 `main` 开始。H01–H08 是归档旧实现的经验，不是当前代码的待修清单；H09 的环境配置已修复。

## 快速定位

- 任务：全文表型位置识别、HPO 标准化、指定患者表型归属。
- 约束：模型不超过 10B；HPO v2026-06-23 的 HP:0000118 分支；禁止外部人工标注表型数据集。详见[比赛资料](docs/competition/README.md)。
- 当前没有新基线实现。旧 CPU 复现及全部旧产物仅保存在 [归档](archive/legacy-20261023/ARCHIVE.md)，不作为新实验默认起点。
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

短信验证后，阿里云页面还可能需要确认已绑定账号；云账号页面跳转成功不代表比赛可提交。本次已核验天池 `getSession` 返回 `login=true`，但比赛详情为 `signed=false`、`showSubmit=false`、`showStsSubmit=false`；页面提供“立即报名”。

**避免方法**：分别核验天池会话和当前比赛/赛季权限，再决定是否提交。保存仓库外 profile 与受限的 storage state，并测试恢复；每次实际提交前复查，不能承诺永久登录。证据/状态：2026-10-23 独立 context 恢复后仍为已登录，当前账号尚未报名；没有提交预测。恢复方法见 [会话说明](docs/competition/SESSION.md)，最新状态见台账。

## H11：认证请求报错可能带出请求头，直连也不等于浏览器网络路径

本环境 Playwright `APIRequestContext.get` 访问天池时出现 IPv6 `ENETUNREACH`，其默认异常详情会包含请求头；已登录浏览器页面内的 `fetch` 则可成功读取同一接口。

**避免方法**：优先使用当前天池页面的会话请求，设置超时并只返回白名单状态；捕获异常后仅输出异常类型，不打印完整堆栈、请求头、Cookie 或会话响应。证据/状态：2026-10-23 已用页面内请求完成登录和比赛权限核验；原始网络错误已绕过，认证内容未写入仓库。

## 更新方式

新问题使用稳定编号，写清触发条件、影响、避免方法及证据。修复时更新该条状态并追加验证版本，不删除防复发经验。待核实猜测不得混入“已确认”；大量案例移入对应实验或审计记录，只留链接。
