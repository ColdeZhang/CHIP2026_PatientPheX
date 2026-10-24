# 边界与复用

- 本线不改Grounded generation或parser。合法数组中的无效quote/重复只拒绝该item，沿原G语义；不能带入已关闭Copy的整行strict弃答规则。
- 新Sap是train32 F2F3监督适配编码器。public41099仅指别名库来源，不能将新权重标成public-only/no_task_fit；新policy及manifest必须明确task_adapted。
- 旧cos≥.90和distinct-concept margin≥.03保持，但新表征尚未校准，这是一项联合控制。支持门不含旧G，避免未变ID借自身旧源获得互证。
- 复用原始254必要query文字及strict-decisions，禁止复用旧semantic向量/scores；不只处理旧mapped项。
- 所有远端启动必须分别冻结admission有效期和计算超时，UNKNOWN只恢复原意图，见主HANDOFF H69。

- S的真实继承B来自`public-new-lgb-support-control/.../supported.pred.jsonl`（471），不能拿实体池的`lightgbm.nested.jsonl`冒充；准备guard在读gold/模型前阻止错误引用，已改用实际既有manifest。
- 默认本地python没有tokenizers，CPU准备固定复用semantic-neural venv（tokenizers0.22.1），254 Rust BERT IDs完整保存；target0.21.4的HF全IDs一致性仍是正式任务门，不凭版本名假定相同。
