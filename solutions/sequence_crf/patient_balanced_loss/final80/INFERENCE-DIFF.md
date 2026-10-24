# α=.5 final80 推断源码准备

入口为 `final_infer.py`，以原 F1 `patient_first/infer.py` **d6055341…** 为主体；同目录 `final_parser.py` **9a9cc3d8…**、`patient_first.py` **25802e68…** 均逐字节复制原件，`raw_store.py` **d52bb0c1…** 逐字节复用已核原始证据发布函数。仅用 full64 `final_infer.py` **baf06014…** 的三收据绑定和 raw先保存机制；不引用其 limited_codec、partial/abstain、额外 logits processor 或64行数据。当前只有源码及静态差异核对，未 import/执行入口、模型、tokenizer、远端或评分。

原模型生成及解析政策保持：batch1、greedy/nonthinking、完整BF16 text model/SDPA、14000输入上限/1600新token、原native GDN无FLA；按 `(input_tokens,row_id)` 排80条，模板、input IDs、模型和adapter参数数目检查不变。`model.generate`、apply_chat_template、encode、decode和sort调用的AST与d605完全一致；没有增加第二次生成、seed、截断、补答案或格式修复。仍只去掉实际末尾EOS后交原strict whole-JSON parser；要求唯一末尾EOS及完整PID/CID，没有partial评分。

改变仅为新训练身份及证据顺序：固定新trainer **3eab3708…**、原train **14af86cf…**、epoch2/32steps、新objective `.5` 与hook **7b9900fc…**；原A800 F1 Torch2.7.1与本A100 Torch2.6.0+cu124差异明确写入fingerprint，不声称逐位等价。主源的 `TRAINING_COMPLETE_SHA` / `TRAINING_CHECKPOINT_SHA` / `TRAINING_DURABILITY_SHA` 当前均为明确 `UNBOUND_ACTUAL_…` 非64hex占位；main在读取任何运行artifact或导入Torch前拒绝。实际完成后必须绑定原三份真实收据、重新冻结源码/计划，不能用旧模型或占位SHA启动。

durability门核原final epoch002物理目录、source/fingerprint、32steps、完整adapter与Adam/RNG成员清单、逐文件fsync及重开SHA收据；实际adapter两文件仍重新hash。完整训练state沿准确持久收据证明，不反序列化或重复读取大Adam/RNG字节。显式持久root及祖先链与输出目录同时绑定。

每条实际生成顺序为：**完整suffix IDs及EOS/cap实际状态持久保存 → 原decode → 原record持久保存 → 原parse_final → strict complete decision**。生成或decode失败保留已取得的证据；解析失败先写rejected decision/失败引用再抛错停止，既不补负答案也不改成abstain。只在80条全部strict通过后发布最终 `responses.jsonl`；每条generation/raw/decision和最后三个索引均exclusive/fsync/重开验证。失败时不会出现可冒称完整的responses文件，也不自动重试或用缓存补齐。

本地精确输入见 `input-sources.json`：原80 prompts **32b52a2d… / 1092467 B**、audit **7fc89abd… / 59512 B**、模型manifest **74e82bd3…**。本次只核原字节，未重新tokenize或读取heldout答案/旧预测。未来运行时仍执行原80 native输入audit，不使用新准备结果替换原合同。

未来控制器计划名可用 `final80-plan.json`：`phase=infer`、`budget_seconds=600`、`grace_seconds=30`、`expected_rows=80`；`files`是准确sources/inputs/三actual receipts的绝对路径→bytes/SHA，`argv`保原CLI加 `--training-durability`，`environment`绑定分配GPU和持久root。输出必须是独立空prediction目录中的 `responses.jsonl`，源与控制文件放目录外。终点核80 fresh/0cache/strict全部通过、全部raw/generation/decision及SHA、driver终态和释放。root已授权的是最终epoch2持久后的这一次600+30；本目录不含已绑定收据的可执行计划，也未部署或启动。
