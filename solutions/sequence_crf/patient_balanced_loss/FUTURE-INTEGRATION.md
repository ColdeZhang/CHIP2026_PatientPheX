# 未来 trainer 集成边界（仅源码设计）

本页是原接线设计；当前已新增probe专用入口并获唯一240+TERM30工程授权，尚未执行；**仍没有混合目标正式训练入口或formal fit授权**。本页依据原 train32 trainer `a3ec17f5…`、selected-CE hook `dfd0759f…`、其 full64 安装位置 `1203139e…:540` 及当前 `mixed_loss.py` `dc98bb97…`。只复用 full64 的实例 hook 方式，不迁入其265行、68步或warmup4配置。

1. **先绑定原行。** 在原 `encode_all` 完全通过后，以原 index、row_id、native IDs/labels/rendered 四SHA、长度绑定已冻结 `row-weights`。原 batch1、labels、attention mask、右侧 padding、124行顺序和6144上下文保持；运行时再次核 `labels != -100` 的完整位置恰为 metadata 的q。P=87、S=16、n_p来自实际准备产物，不从预测或正例数推导。

2. **每个 microbatch 一个不可变闭包。** 在原循环的 `model(**batch,use_cache=False)` 前，捕获本行 metadata 与实际 `g=len(group)`，附 epoch/group/microbatch/row身份；不用会被下一行覆盖的全局 `current_row`。只给实际 Qwen base 实例安装 `loss_function` 适配器，保存原 callable 并核所有权。适配器核 `vocab_size`、`ignore_index=-100`、`num_items_in_batch=None`、未预先 shift 的原 labels；未覆盖的参数分支直接拒绝。随后调用既有 `mixed_causal_loss`，不改其公式。

3. **作用域覆盖 forward 与 backward。** 闭包持续有效直到本行原 `(loss/g).backward()` 返回，保留 `use_reentrant=False`；检查点重算如再进入 loss，也只能看见同一不可变绑定，不能在 forward 返回时就清空。每次仅一行在途，禁止线程并发、异步排队或把多行 forward 后再 backward；不靠新增逐行 CUDA同步替代正确绑定。`try/finally`核当前hook仍属本行、恢复原callable并清空上下文；异常立即停，禁止带残留hook继续。下一行、保存checkpoint及退出前均核无活动上下文。

4. **维持原计算与更新路径。** 仍是一次原模型 forward、完整 BF16 LM head/GEMM，再按q−1选择监督logits并做FP32 CE；不投影选中hidden、不改 `logits_to_keep`，不再跑第二次模型。原 CE一半保留格式/key/EOS/换行；新增值项已含 `g*S/(P*n_p*n_value)`，外层原 `/g`只做一次，尾4不再额外除4。保留原r16/LoRA alpha32、AdamW、clip、scheduler、warmup2、两轮seed20261023/24与每轮15×8＋1×4、32步及原checkpoint/RNG流程；新objective/metadata/hook SHA加入独立fingerprint，不能冒充旧训练状态。

5. **仍待真实执行证据。** CPU小张量合同不证明检查点重算下的模型接线、真实显存、LoRA更新或质量。后续先冻结具体trainer差异与单次hook生命周期检查，再等root明确分配GPU、探针/正式预算及终点；本页不提供启动命令、不扩大授权。任何实际误差先持久保存再断言，不能用设计说明代替GPU合同。
