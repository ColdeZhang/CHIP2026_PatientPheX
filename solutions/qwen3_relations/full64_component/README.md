# 原生 Qwen3 全量融合组件与共享训练接口

本次实例由 [plan.json](plan.json) 固定：265 行、2 轮、68 次更新、尾批 1、warmup 4；仅为 root 已验证的固定 2of3 生成完整组件。旧独立保护并集的 F0 负结论仍保留。本实例不读 originaldev gold、不独立评分。

固定执行入口是 `driver.py --plan plan.json`，远端唯一控制器依次管理 probe 240 秒、fresh formal 2400 秒、infer 1200 秒，每阶段 TERM 30 秒。失败停止；未知连接状态先恢复原 job。接手信息见 [运行恢复入口](../evidence/full64-component-resume.json)。

## 可复用接口

同架构的新任务可以从独立入口导入这些函数；`native.main` 仍绑定本实例数据身份，不应直接拿去运行别的任务。

- `common.load_model(plan, output, adapter=None)`：固定公共 BF16 Qwen3、FP32 r16 LoRA；返回 Torch、NumPy、模型、可训练参数映射、实际模型 manifest。
- `native.epoch_groups(plan, epoch)`：按 `seed + epoch` 生成完整行序与累积组，返回 `order, groups`。当前累积组固定为 8；尾组按 `recipe.last_batch_size` 检查。
- `native.probe(plan, output, cpu, torch, model, named, model_manifest)`：初始 8 行→真实 Adam→3 行累积→全部实际 Pareto。保持已有优化器和梯度压力，旧损失合同不重跑。返回的历史状态名包含 FULL64，调用方应另外准确声明自己的任务规模；不得以名字代替真实覆盖证据。其他任务的生成检查由其独立入口接续，最终丢弃探针状态。
- `native.train(plan, output, cpu, torch, np, model, named)`：复用原完整训练 loop，按实际组大小平均 CE；每轮持久化适配器、Adam、scheduler 和 RNG，并重开比对与 fsync。逐值重开不等同中断续训等价。

关键配置：`recipe.rows/epochs/updates/updates_per_epoch/last_batch_size/warmup_steps`、`seed`、`lora`、`budgets.formal`、`durability_root`；模型/运行库/资产和源码身份仍须按该任务固定。学习率 1e-4、weight decay .01、clip 1、累积 8 维持原配方，不能通过只修改配置假称已更改函数内的固定参数。

`cpu.counts.train` 必须含 `encoded`（path/bytes/SHA）、`tokens`、`supervised_tokens`；每行含完整 `input_ids`、等长 `labels`、稳定 `row_id`。probe 另需 `stress_initial8/stress_accumulated3/pareto_indices/pareto_shapes`，且 `plan.probe_pareto_shapes` 必须固定同一组真实形状。265/34/68/尾1 与 87/11/22/尾7 的实际 schedule 覆盖已做 CPU 合同；这不替代其他任务的分词、目标或真实 GPU 工程验证。

`native.infer/parse_complete` 当前固定64行和闭集 CID/PID 完整 JSON；canonical-name、NIL 等不同目标须用自己的完整输出解析与约束推断入口，不套用该解析器。

本次原生 forward/standard CE 语义沿旧 F1 实证复用。实际新形状 GPU 已通过；完整 full64 训练与64预测是否完成，应看真实回执，不能从共享接口或探针通过推断质量。
