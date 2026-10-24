# 第八阶段：官方反馈后的并行扩展

用户回传 2026-10-24 15:54:01 的最高901提交结果：官方Score **.6922、排名第五**；men .7510、doc .8124、micro .6499、macro .5555，第一名 .7389。来源是用户消息，没有再次访问官网；原不可变接收凭据保持，见[独立结果记录](../submissions/round2-best-20261024-user-result.json)。对应原dev .754361398，差.062161398。两关系分项分别差.088880/.101686，约占总落差76.6%；不能将单次差距外推为本地.8必定超过第一名。

用户明确继续固定原dev≥.8，同时允许扩展方法、多分支/worktree、多代理迭代。每个独立模型≤10B，允许集成；按原文献隔离和公共v2比较，禁止将dev标签作为训练资源。新方法的价值以完整验证与互补贡献判断，不以是否原创判断。官方上传仍暂停。

| 新工作线 | 独立分支 / worktree尾目录 | 首要机制与职责 |
|---|---|---|
| span_normalization | explore/span-normalization / explore-span-normalization | 调研并实现跨度检测与HPO规范化联合学习或显式span方法；区别既有BIO/GLiNER简单追加，不只换encoder名称 |
| document_relations | explore/document-relations / explore-document-relations | 调研ATLOP/DocRE等成熟文档级关系建模，联合患者及表型多处提及；区别已失败的局部pair CE/简单max-MIL |
| grounded_generation | explore/grounded-generation / explore-grounded-generation | 调研有原文证据的患者表型生成与独立规范化；明确区别旧8B开放HPO编号失败，优先复用9B及规范化资产 |

每线维护自己的WORKLOG/experiments/NOTES/evidence，先用已有历史排除重复，随后完成可运行初版和训练内完整对照，不停在调研报告。允许复用成熟代码、模型和现有已核资产；源码/许可/版本保存。常规工程修复自主处理，未知状态恢复原作业，避免重复启动。模型/大型raw留外部，关键结果及时commit/push并核远端。

主线程负责共享数据/评测来源、GPU调度、跨线比较和融合。既有全文indexed与FLA完整质量继续收口；患者α.5已完整负：primary .661496611109低原 .663841820660，alone .636714244169，不扩大/不扫α，终审进行中。新路线CPU/来源准备并行，GPU先盘点空卡再按实际成本运行，不抢现有作业。强方法若在固定训练内留出上改善，则验证另一折、训练full64并在原dev作最终验收；新训练输入上限须实际编码核验，不能截样本或套旧cap。

本阶段无新best，原dev仍.754361398；新分支未达标前不合并实现，主分支持续保存统筹和真实进展。

三个新worktree已从main `0c0672d` 创建，对应远端分支均已push/ls-remote核同SHA。span负责人 `/root/span_normalization`、docRE负责人 `/root/document_relations` 已启动；生成负责人 `/root/sequence_recover_0518` 正将旧α负实验收尾后转新worktree。span拟先显式起止联合打分+固定规范化，docRE独立选择完整文档多提及机制；尚无新GPU或新分数。

首轮资源排程：span与docRE复用.3已有Biomed/BioLink/Sap资产，span先获≤600+30窗口（尚未实际启动），释放后docRE接续；两者CPU准备并行。A100 GPU2拟留给患者证据生成，当前仅旧α小补包I/O且连接暂退避，不等同GPU作业。A800新全文80与.100 FLA80原质量运行不受影响。显式分类器保存紧凑全候选分数便于后续学习融合，生成器保留raw/IDs与解析状态；不把大张量/日志放Git。

启动前统一划分：三新路线初版均使用F2+F3 train32→F1完整16篇/39患者，第二折再train48→F0。span/docRE原先拟F0方案尚未实际GPU或数据准备，已在启动前协调修订，方法/预算不变。公共冻结模型可复用，任何监督别名、候选/特征模型仍须核本折来源；不把强train48对照称同训练量对照。三线各自固定非目标对象并报告分项/差值，root在相同F1文献上组装完整融合，避免为对齐重复补训。

GP首个实际训练已09:39:09.637056在.3 GPU0唯一启动proc_03655e70-29ef-4447-9413-84f06e920184/PID1196540，总600+30含丢弃工程probe、fresh8epochs、完整588窗与声明池Sap规范化。train32 1148窗/117170tokens，2266跨度有2263可精确对齐，3条不可表示披露；嵌套仅10，不夸成主要缺口。生成池固定logit≥−2、最终决策>0且不扫阈值，完整声明池分数外存。docRE及生成线继续CPU准备，尚无新质量分数。

GP首入口6.766秒在任何forward前因未列已弃用pooler两权重退出，随后.3短暂离线；原失败完整保存。09:45:05.190878连接恢复后唯一proc_f83b21dd/PID1196954继续同原09:49:09.705568+30截止，未重置预算，数据/方法不变。docRE完整CPU实际1.250秒，train300窗/held156窗；727正对中720有attention支持交叠，7条context=0但两端全提及embedding保留且不删pair，不冒称完整全局注意力。释放后最长doc工程获120+30窗口。生成任务已澄清quote为复制的表型词面、非人工患者证据：87训练451672tokens、监督11884、max输入9210、max监督610；39held max10080。全部身份/targets保留，原生CPU回放120+5及A100 GPU2真实14反向/Adam/三Pareto420+30已授权，尚未新训练。
