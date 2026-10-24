# 方法笔记

本路线研究已预训练关系头和schema驱动边界模型的增量，区别旧GLiNER实体头与from-Biomed DocRE。完整预训练样本谱系未公开，不声称纯架构因果。

根线程来源见main docs/research/multitask-relation-readiness-20261024。GLiNER2上游2.1.0通过AutoExtractor按boundary加载，旧GLiNER2类仅span，不能混装。公开loader按strict state_dict加载。max_len为word-token相关接口，真实wordpiece/文本覆盖须通过实际processor确认，不能据4096配置承诺完整报告。

全候选与原分数外存；患者头必须映射原input明确mention坐标/身份，不能用训练或heldgold补病人。规范化固定strict/Sap政策，未知结果保持拒绝并计数。

- ModelScope本次weight响应在524288000B结束，目标774366564B，必须核size/SHA并Range续传；HTTP请求结束不等于权重完整。证据R/stage8-schema-extraction/assets-failure-inspection.json。
- 远端默认python命令不一定存在；使用已核的绝对解释器。.100通用ScratchVL解释器本身没有transformers；需显式已核私有PYTHONPATH，不能把另一路的私有包当全局已安装。
- JointIE原生processor会为不以句末标点结束的窗口追加一个句点；审计记录这一步，caller映射只接受真实原字符。使用transform_record（不是旧transform_and_format）获取当前公共单记录合同。collate默认fallback改为error_policy=raise，防止静默dummy记录。
- declarative relation description目前未被上游compiler写入model_schema；模型实际只见关系名称与head/tail槽位。不能用文档中的自然语言宣称模型具有未实现的硬性语义约束。
- .3→.100首次server启动返回UNKNOWN_OUTCOME，随即.3 LUNA_OFFLINE，动态PORT未回收，不能猜port或重复启动。root协调由span稀疏恢复.3，我停止并发轮询；改为.100独立同SHA资产传输，.100旧fetch从未启动，不竞争同输出。旧server限时180秒、外层210+30，原意图/token保留待核终止后清理。

后继可学习适配（尚未训练、未作新质量）：上游 `InputExample`/`Relation` 虽类型注解为字符串，实际 `_is_locatable` 接受 `{text,start,end}` 并转 `SurfaceSpan`，processor `_resolve_span` 使用明确坐标；因此可以直接利用train32真实患者mention、表型span及文档association，避免同名字符串全局回搜。训练关系仍需声明从document-level关联到同窗mention对的弱监督假设，并测覆盖。stock `dedupe_contained_entity_spans=True` 会过滤同类嵌套gold；若做精确span监督应显式False且核全部gold。`max_gold_per_query`/relation pair budget必须先做train32 census，不通过截断gold制造训练成功。这些是后继机制与工程约束，不改变当前冻结zero-shot首版。

纯输入覆盖：native384/64共164窗中80含给定患者mention，39患者都被覆盖；其余84窗本版不能推断跨窗患者关系。统计源R/stage8-schema-extraction/prepared/patient-window-census.json，未读heldgold。
