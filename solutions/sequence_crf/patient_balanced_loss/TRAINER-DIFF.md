# 固定 α=.5 trainer 接线草稿

仅新增 [train_patient_balanced.py](train_patient_balanced.py)，机械复制原 [train_patient_first.py](../patient_first/train_patient_first.py) `a3ec17f55ca7007ab1f41879ff07e15586f78d7b3064a584464cd90fa935164e` 后接线。**当前仅 SOURCE 草稿，未 import、执行、部署或训练；没有正式运行计划、启动命令或预算。** 本轮只做 AST、文本差异及静态来源核对，未重跑既有 CPU 合同。后续正式 GPU 预算由 root 另行确定；既有数值合同或工程 probe 不代表本 trainer 已运行。

| 范围 | 原逻辑与本次差异 |
|---|---|
| 数据与编码 | 原 `training_rows`、`verify_sources_and_data`、`encode_all`、`collate` 原样。保留 train32 的124行、32文献、87患者、400614完整 tokens、4694监督、max5556；6144仅拒绝上限。原 input IDs、labels、mask、完整序列及顺序不变。新增绑定函数把每一原 index/row_id/pmc_id 与完整 row metadata、原 native audit 对齐；input IDs/labels 用原默认 `json.dumps` 空格计算 SHA，连同 rendered prefix/full 四SHA全部一致才继续。 |
| 唯一目标差异 | 固定 `mixed_loss.py` `dc98bb97…` 与接口恢复 `RowLossScope` v2 `7b9900fc…`，不编辑 core/hook。混合系数 `.5` 独立于原 `CONFIG['alpha']=32`（LoRA alpha）。P87/S16及既存 n_p/value positions 来自原准备；不重新分词切值、重计患者、按正例加权或删空列表。 |
| 一次 forward/backward | 每原 batch1 捕获完整 metadata、epoch/group/microbatch/index 与真实 `g=len(group)`。捕获一次 `original_loss=base.loss_function`，核原生 `ForCausalLMLoss`，把原 `fixed_cross_entropy` 传给 core。`with RowLossScope` 覆盖原 forward、finite 检查及原 `(loss/g).backward()`；检查点重算期间仍持有同一绑定。下一行、optimizer、epoch保存与退出前核 callable 恢复及无活动 scope。 |
| 计算路径 | 完整原 BF16 LM head/GEMM 保留，没有选 hidden 投影、第二次模型 forward 或 `logits_to_keep`。仅监督位置 q−1 选 logits 后 FP32 CE；原 token CE 半项覆盖格式/key/EOS/换行，新半项含 `g*S/(P*n_p*n_value)`，外层只除一次 g。尾组实际 g4，其余 g8。 |
| 原优化与顺序 | 原 r16/alpha32/dropout.05、AdamW/lr1e−4/weight decay.01、clip1、warmup2、scheduler32及 nonreentrant checkpointing 保留。原 batch_plan 及 seed20261023/24 每轮15×8+1×4、两轮共32次 optimizer/scheduler调用不变；与冻结 epoch-groups 逐组对照。未迁 full64 的265行/68调用/warmup4，也未迁 probe 无warmup一步策略。 |
| 完整 epoch 状态 | 原 epoch staging、adapter safetensors/config、`training_state.pt` 中 Adam/scheduler/Python/NumPy/Torch/CUDA RNG及文件 SHA清单原样。仅将原单独 rename 调用换成下述原协议的durable发布函数。原暂停、完整 epoch-only恢复、部分 epoch重放语义保留。新scope观察放各次 control身份独立目录，避免重放覆盖旧观察。 |
| 语义身份与旧状态拒绝 | 新 fingerprint schema 绑定本源、原 a3ec 来源、准备manifest、row-weights、epoch-groups、mixed/hook/native loss 源 SHA、固定objective及实际环境。旧objective在任何本轮输出写入前即拒绝；本目标恢复仍需原完整 fingerprint相等。不能把旧LoRA/Adam状态、probe丢弃状态或复制过来的旧checkpoint当新目标训练。 |

新增三个必填 CLI 字段，继承原其余 CLI，不提供运行实例：`--row-weights` 指向原 `row-weights.jsonl`（308095 B，`e1e4cc5e…`）；`--weights-preparation-manifest` 指向已完成准备 manifest（5074 B，`c713111b…`）；`--epoch-groups` 指向原双epoch合同（594436 B，`6b80ef9d…`）。三件的完整 SHA及大小固定在源中；路径可显式绑定未来部署副本。`mixed_loss.py` 位于 trainer 同目录，新hook位于相对路径 `interface_recovery/row_loss_scope_v2.py`，均核完整SHA，源码快照保留该目录结构；原 Qwen helper/protocol/schema、native CPU receipt、数据与模型来源检查保留。

相对首份 trainer 草稿 `7caedb5f…` 的机械接口补丁仅为：hook SHA/路径改为v2、构造函数末尾传 `observer`、observer在任何参数guard及loss前将原实际dict追加到本次control独立 `actual-loss-calls.jsonl` 并逐文件/祖先fsync；两diagnostic只接受None，未知键/非None仍拒绝。原loss公式、actualg及forward/backward作用域不改。独立纯Python接口合同在源码推送 `c785f246…` 后唯一运行通过：27case/23个新observer、0.216996秒、rc0，result `d76b7da8…` / terminal `043ba6cd…`，见 `R/research/patient-balanced-mixed-loss-alpha05/interface-recovery/python-contract-v1`；真实PEFT/Qwen/decorator函数body仅与fake对象运行，无Torch/实际target/数值合同重测。这不是trainer、模型backward或GPU通过；原hook97a0与原GPU失败均保留。

原 F1 实际 Torch2.7.1，本草稿要求已做数值合同的 A100 Torch2.6.0+cu124、TF5.20.2/PEFT0.21.2，并核 native `loss_utils.py` `83db16b2…`。运行库差异与 selected-CE 数值等价工程会写入 fingerprint；不声称跨Torch逐位训练等价，不把已有合成 logits 合同当真实 LoRA/Adam 梯度因果或质量证据。

**持久发布协议现已接入源码，尚未执行，不能宣称取得持久 ACK。** 按父线程明确范围，把原 [full64 trainer](../patient_first_full64/memory_recovery/formal/trainer.py) `1203139eabbbb501d4e532ea9080773b9f662a8aae9638f4519b0f58a24109f7` 中 `durability_fsync_directory`、`durability_inventory`、`durable_publish_checkpoint` 复制到本 trainer；原两辅助函数字节一致，发布函数只把 `global_step == 34*completed` 的检查常量改为16、错误说明 original265改original124。原epoch存储块只将 `staging.replace(destination)` 换为 `durable_publish_checkpoint(output,staging,destination)`，没有更改序列化、Adam/RNG、124×2、32calls、warmup2或尾4。

三个函数保留原安全路径/无symlink/完整成员清单、逐文件与深层目录fsync、rename、至显式 `PATIENTPHEX_DURABILITY_ROOT` 的祖先fsync、重开SHA、exclusive receipt及receipt父目录fsync。它们在本trainer中运行，因此原 `sha256(__file__)` 正确绑定本trainer实际源及checkpoint fingerprint，无独立exporter身份混淆。每个epoch只有收到原协议返回后才继续写progress/paused/complete；post-rename SHA失败保留目录但不会有成功ACK。显式持久根路径须由未来正式方案绑定，不能用目录存在替代收据。

本次只做AST/文本差异：两辅助函数相同；发布函数AST仅有34→16及265→124字符串两处常量差异；原checkpoint创建/保存块除单一发布调用外相同。没有import/执行trainer、文件模拟测试、317项数值合同或模型。原 `validate_checkpoint` 的完整epoch/state检查也原样，未额外实现ACK恢复校验；未来正式控制器仍须核两份逐epoch ACK与checkpoint实际SHA，并回收最终terminal/output清单。正式预算依真实probe结果由root另定，本次无formal运行计划或命令。
