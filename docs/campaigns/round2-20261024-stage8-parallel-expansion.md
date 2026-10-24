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
