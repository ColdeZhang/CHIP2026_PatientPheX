# 原Grounded引文使用监督适配Sap：CPU准备

**已完成无gold/无模型的准备，正式查询尚未执行。** 预锁唯一终点为 `B4 ∪ [Gadapted ∩ (S∪Q∪C)]`，支持门不含旧G。保持原普通Grounded32 d52输出，完整39行中的原EOS/JSON及逐item无效/重复拒绝不变；不是已关闭Copy受限生成的继续实验。

原完整提取有573合法原文anchors，复用全部254必要查询（1538 native tokens，最大15，0超32），包含原semantic阈值弃答项。旧strict结果/原文坐标已逐条核对，不重生成9B，不重算训练别名，不复用旧semantic向量或分数。

| 纯预测支持统计（不是质量） | 关系数 |
|---|---:|
| B4 | 524 |
| 旧S存盘 | 505 |
| S继承B，全部在B4中 | 471 |
| 旧pure Q | 416 |
| 旧C | 395 |
| S∪Q∪C | 595 |
| 支持门在B4外 | 71 |

新权重是同F2F3 train32、3epoch/219更新的固定最终state `b0482af6…`，新41099公共别名bank为 `fa312ba7…`；公共别名顺序不变，权重必须标`task_adapted=true/no_task_fit=false`。真实加载为FP32参数＋CUDA FP16 autocast，CLS转float后torch L2；不能改成model.half。新[policy](policy_v1.json)和[bank证书](bank-manifest.json)不沿旧public权重身份；[owner证书引用](evidence/model-binding.json)另绑定实际来源。

CPU合成合同覆盖：原exact/NIL/歧义保护、短缩写、.90阈值/.03不同概念间隔、共享别名导致的概念并列、缺查询证据拒绝、超32不截断、旧G不自证、逐item拒绝及非关系字节保护；过期/未授权入口拒绝。全部通过，未加载Torch模型或评分。原阈值在新表征中未校准，仍可能损害质量；不据Task1路线的后来结果修改本预锁公式。

[查询草案](query-plan-draft.json)：现成.3资产无需传1.309GB state或126MB bank，唯一254查询，建议180+30秒；60秒是保守计划代理，不是实测。父任务同host完整219更新＋4次bank＋303held共50.879秒；新任务还需state核验/加载及CPU全概念排序。实际原生HF254IDs、checkpoint加载/输出格式必须在正式有界任务中核，不由CPU合同冒称通过。

后继由root分配GPU并绑定真实release/fresh门、300秒启动有效期和独立180+30运行时钟，sourceplan先push，再唯一启动；UNKNOWN恢复原意图。查询全量原件通过后沿[finalize.py](finalize.py)与[assemble.py](assemble.py)固定语义，先全16/39/1189、保524旧关系freeze/push，再最多1新v2/旧0（byteexact0）；F1严格正才固定F0。当前controller/正式执行计划/collector及score入口待该授权阶段绑定，未启动任何远端查询。

CPU复现：`CUDA_VISIBLE_DEVICES='' /workspace/patientphex-runs/day1-20261023/semantic-neural/venv/bin/python solutions/adapted_quote_linking/contracts.py`。合同使用独占写入，已有成功证据时先读证据而非重跑覆盖。全量准备产物在仓外 `research/adapted-quote-linking-v1/F1-prepared-v1/`。
