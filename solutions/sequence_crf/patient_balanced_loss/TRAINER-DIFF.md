# 固定 α=.5 trainer 接线草稿

仅新增 [train_patient_balanced.py](train_patient_balanced.py)，机械复制原 [train_patient_first.py](../patient_first/train_patient_first.py) `a3ec17f55ca7007ab1f41879ff07e15586f78d7b3064a584464cd90fa935164e` 后接线。**当前仅 SOURCE 草稿，未 import、执行、部署或训练；没有正式运行计划、启动命令或预算。** 本轮只做 AST、文本差异及静态来源核对，未重跑既有 CPU 合同。后续正式 GPU 预算由 root 另行确定；既有数值合同或工程 probe 不代表本 trainer 已运行。

| 范围 | 原逻辑与本次差异 |
|---|---|
| 数据与编码 | 原 `training_rows`、`verify_sources_and_data`、`encode_all`、`collate` 原样。保留 train32 的124行、32文献、87患者、400614完整 tokens、4694监督、max5556；6144仅拒绝上限。原 input IDs、labels、mask、完整序列及顺序不变。新增绑定函数把每一原 index/row_id/pmc_id 与完整 row metadata、原 native audit 对齐；input IDs/labels 用原默认 `json.dumps` 空格计算 SHA，连同 rendered prefix/full 四SHA全部一致才继续。 |
| 唯一目标差异 | 固定 `mixed_loss.py` `dc98bb97…` 与 `RowLossScope` `97a0c5ce…`，不编辑 core/hook。混合系数 `.5` 独立于原 `CONFIG['alpha']=32`（LoRA alpha）。P87/S16及既存 n_p/value positions 来自原准备；不重新分词切值、重计患者、按正例加权或删空列表。 |
| 一次 forward/backward | 每原 batch1 捕获完整 metadata、epoch/group/microbatch/index 与真实 `g=len(group)`。捕获一次 `original_loss=base.loss_function`，核原生 `ForCausalLMLoss`，把原 `fixed_cross_entropy` 传给 core。`with RowLossScope` 覆盖原 forward、finite 检查及原 `(loss/g).backward()`；检查点重算期间仍持有同一绑定。下一行、optimizer、epoch保存与退出前核 callable 恢复及无活动 scope。 |
| 计算路径 | 完整原 BF16 LM head/GEMM 保留，没有选 hidden 投影、第二次模型 forward 或 `logits_to_keep`。仅监督位置 q−1 选 logits 后 FP32 CE；原 token CE 半项覆盖格式/key/EOS/换行，新半项含 `g*S/(P*n_p*n_value)`，外层只除一次 g。尾组实际 g4，其余 g8。 |
| 原优化与顺序 | 原 r16/alpha32/dropout.05、AdamW/lr1e−4/weight decay.01、clip1、warmup2、scheduler32及 nonreentrant checkpointing 保留。原 batch_plan 及 seed20261023/24 每轮15×8+1×4、两轮共32次 optimizer/scheduler调用不变；与冻结 epoch-groups 逐组对照。未迁 full64 的265行/68调用/warmup4，也未迁 probe 无warmup一步策略。 |
| 完整 epoch 状态 | 原 epoch staging、adapter safetensors/config、`training_state.pt` 中 Adam/scheduler/Python/NumPy/Torch/CUDA RNG、文件 SHA清单及 rename 原样。原暂停、完整 epoch-only恢复、部分 epoch重放语义保留。新scope观察放各次 control身份独立目录，避免重放覆盖旧观察。 |
| 语义身份与旧状态拒绝 | 新 fingerprint schema 绑定本源、原 a3ec 来源、准备manifest、row-weights、epoch-groups、mixed/hook/native loss 源 SHA、固定objective及实际环境。旧objective在任何本轮输出写入前即拒绝；本目标恢复仍需原完整 fingerprint相等。不能把旧LoRA/Adam状态、probe丢弃状态或复制过来的旧checkpoint当新目标训练。 |

新增三个必填 CLI 字段，继承原其余 CLI，不提供运行实例：`--row-weights` 指向原 `row-weights.jsonl`（308095 B，`e1e4cc5e…`）；`--weights-preparation-manifest` 指向已完成准备 manifest（5074 B，`c713111b…`）；`--epoch-groups` 指向原双epoch合同（594436 B，`6b80ef9d…`）。三件的完整 SHA及大小固定在源中；路径可显式绑定未来部署副本。两损失源码须位于 trainer 同目录并核完整 SHA；原 Qwen helper/protocol/schema、native CPU receipt、数据与模型来源检查保留。

原 F1 实际 Torch2.7.1，本草稿要求已做数值合同的 A100 Torch2.6.0+cu124、TF5.20.2/PEFT0.21.2，并核 native `loss_utils.py` `83db16b2…`。运行库差异与 selected-CE 数值等价工程会写入 fingerprint；不声称跨Torch逐位训练等价，不把已有合成 logits 合同当真实 LoRA/Adam 梯度因果或质量证据。

**保存完整状态不等于已取得持久 ACK。** 原 a3ec 的写文件、SHA与目录 rename 本身未完成服务器逐文件及全部祖先 fsync/重开SHA证明；本草稿没有顺手更改 checkpoint 实现，也不以目录存在冒充 durable。未来正式执行前仍须由独立 exporter/控制器明确绑定逐epoch及最终实际 SHA、文件/目录/祖先 fsync、重开核验及持久收据，可沿已核 [full64 durable_publish_checkpoint](../patient_first_full64/memory_recovery/formal/trainer.py) 的恢复协议；不能迁其训练规模或据此声称本草稿已持久执行。正式预算、终点收件与 ACK 实现均留待对应正式方案。
