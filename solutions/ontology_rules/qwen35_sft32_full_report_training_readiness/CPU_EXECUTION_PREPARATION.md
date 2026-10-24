# 两个单次 CPU 合同的执行准备

主线程已确认 native token/mask 计数和目标 Torch 的 selected-CE 合成数值合同属于 CPU 可行性授权；由父线程在原 F0 双质量及统一终审之后执行。本文件保存的是**准备完成、尚未执行**的入口。没有新增模型、训练、GPU 探针或质量授权。

| 独立任务 | 本地入口 | 远端私有 namespace | 原始输出 |
|---|---|---|---|
| native token/mask | `run_cpu_once.py` | `ontology-round2/qwen35-sft32-full-report-training-cpu-v1` | `native-result.json`、`native-row-audit.json`、`CPU-complete.json` |
| selected-CE 合成合同 | `selected-ce-cpu/run_cpu_once.py` | `ontology-round2/qwen35-sft32-full-report-training-cpu-v1-selected-ce` | `contract-result.json`、`CPU-complete.json` |

本地入口应从外部 runs 目录使用，不能从缺少压缩输入的仓库副本启动。两者均固定 A800 mount `69b991ee-183f-4add-af1a-579af92479c9`、UID1045、自有 qwen-venv 与 a800-migration runtime；`CUDA_VISIBLE_DEVICES=''`。只调用 CPU，源码不会检查或占用显卡。

控制器沿用已审 F0 `7bbf6392…` 的部署、单次 timeout 和回收结构，针对本次精确清单改入口及结果验证。native 保持原 `cpu-plan.json` 的四件 payload，加 plan 共五件；tar 700,133 字节分 11 个 64KiB 片。selected 三件源码加 plan 共四件，tar 6,618 字节，1 片。每件 SHA、打包成员 basename/类型、完整 tar SHA、上传分片 SHA、实际远端 compile 均需通过。源只在不存在时写入并 fsync；实际远端回执不属于本次已经发生的事实。

两任务各限 **60 秒＋TERM 5 秒**，MCP 调用上限 75 秒。控制器在首次 MCP client 创建前保存本地排他部署 intent，在实际子进程前保存本地及远端排他 CPU intent。传输/返回状态未知时保留原 intent，只用 `--mode inspect` 确认，终态可用 `--mode collect` 回收；不能再次 `--mode run`。inspect/collect 均没有子进程启动路径。native 和 selected 自身还各有排他 intent。重复回收仅接受既有文件原字节相同，不覆盖不同结果。

实际执行命令由父线程在上述时机调用：

```text
python <外部目录>/run_cpu_once.py --mode run
python <外部目录>/selected-ce-cpu/run_cpu_once.py --mode run
```

回收采用最多 48KiB 的压缩包分片，逐片、整包、每个成员 SHA 校验后按原字节落盘。`actual-result.json` 只作对应原始 result 的相同字节副本。包装器的 plan/source/controller SHA 与 rc、60/5 参数也绑定核验。失败时保留原 stderr/stdout 和状态；不能用再次启动来补失去的回执。

native 入口和 plan 原字节保持 `54d32fe3…` / `a85cc269…`。回收额外核原/全文 124 行、32 文献，原 402,594/max5,523、原 encoded SHA `7fa9ef2b…`，两套监督均 7,170；逐行 assistant IDs、发布 native/Auto/template/decode 合同必须通过。

selected 合同使用实际运行库导入的 `transformers.loss.loss_utils`，与已核 `83db16b2…` 的原生源副本逐字节相等；导入 `selected_loss.py` 原件 `dfd0759f…` 并确认其调用的是同一个 native fixed CE。目标固定 Torch2.7.1+cu126 / Transformers5.20.2，不把过去 Torch2.6 的数值结果冒充本轮结果。

- 288 个基础组合覆盖 FP32/BF16、四种 logits 量级、六种因果监督/忽略模式、mean/整数/张量分母以及显式/隐式 shift。
- 保留同形状完整 head，核 hidden/head 梯度；按实际样本均值分别核 1/2/4/8 行累积，包含原124的末组4行。
- 只扰动被忽略位置 logits，要求 loss 与梯度完全不变；额外核自定义 ignore=-7。
- loss 容差 rtol/atol=2e-6，梯度 rtol=2e-5 / atol=2e-7；无有效监督的 native NaN/零和行为保持，不制造零损失成功。

这只是合成 CPU 数值合同，没有真实任务输入、模型权重、optimizer、GPU 或实际训练显存；通过后也不能自动宣布全文训练可行。

已完成的本地检查为源/内嵌远端脚本编译、全部 payload SHA、无启动模式检查，以及两个有区分力的反例：已有 intent 必须在任何 client 访问之前拒绝；已存在但不同字节的回收文件不得覆盖。见 `controller-static-checks.json`。没有执行 native 计数或 selected 数值计算，没有加载 MCP 凭据或调用远端。
