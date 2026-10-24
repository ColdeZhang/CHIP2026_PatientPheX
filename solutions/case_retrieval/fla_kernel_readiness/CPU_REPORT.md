# 私有 CPU readiness 交付（2026-10-24 UTC）

**结论：私有安装与 native CPU 准备通过；纯 CPU 的 FLA 实际绑定未通过，原因是 Triton 导入时要求可见 GPU driver。没有启动 GPU。** 不能把安装成功/native合同通过写成fused已启用。

- .100实际：Python3.10.12、glibc2.35、torch2.7.1+cu126、Triton3.3.1、CXX11 ABI TRUE；磁盘空闲446,046,294,016字节。只读采样时A30卡1 fc5d…为0MiB/0%，卡0已忙；以后必须重新分配/查占用。
- 解释器：`/home/zhangyuheng/projects/ScratchVL/.venv/bin/python`。复用Qwen3.5专用runtime（TF5.20.2/PEFT0.21.2/accelerate1.15.0）以及旧round2/runtime补充regex等依赖，顺序固定在`private-env.json`。单独旧round2/runtime是TF4.51.3，不能拿来替代。
- 所有新增远端内容限定`research/fla-kernel-readiness/private-v1/`；安装仅`target/`，wheel仅`wheels/`。`--no-deps --no-cache-dir --no-compile`，没有改torch/TF/共享site-packages，没有causal-conv大轮子。
- fla-core0.6.0 wheel 1,085,221B，SHA `b0708b70e0b119cd58a0a5146ea56cb175cef6ed969d8ac60267e3e077ce3ebd`；必需einops0.8.1 wheel64,359B，SHA `919387eb55330f5757c6bea9165c5ff5cfe63a642682ea788a6d472576d81737`。本地和远端下载后均按官方PyPI SHA核实。einops原环境缺失，root追加授权后才加到同一私有target。
- 安装实测42.046秒；失败fused CPU导入1.526秒；native合同5.144秒。其它短只读/源码CPU检查没有逐项计时，不将这些部分和冒充完整准备总时长。所有远端命令有限timeout，未开启长期CPU/GPU进程。

## 实际阻断与通过范围

`cpu-fused-import-failure.json`：CUDA_VISIBLE_DEVICES为空时，`fla.ops.gated_delta_rule`先加载`fla.ops.__init__`的ABC/utils，Triton3.3.1 Autotuner构造调用`driver.active.get_benchmarker()`，报`RuntimeError: 0 active drivers ([]). There should only be one.`。没有修改FLA/Triton源码、伪造driver或偷偷暴露GPU。TF装饰器捕获这类异常会native回退，因此导入TF成功仍不能证明FLA绑定。

`cpu-native-contract.json`与`cpu-source-contract.json`已实际通过：

1. TF实际modeling SHA `b4b17fe…`/hub_kernel SHA `a487505…`与官方固定tag相同；两GDN、两conv在CUDA隐藏时均实际native绑定，CUDA前后未初始化。
2. 仅用缓存tokenizer产生纯合成提示；实际长度128/255/256/513，decode续写8 token；每条decode→encode严格回到原IDs。输入SHA `1d57caa3bceaab8b9f024452c5c0a3f25502f62b591b571c1a73f0bccceec4ad`。未读赛事样本/标签。
3. GPU源码只做AST和无launch导入检查；比较函数的零分母/近零绝对门槛、1%通过/3%失败边界实际CPU检查通过。复用的因果loss/忽略位置梯度合同通过，含set的JSON写入实测通过。
4. 模型小配置/tokenizer文件actual SHA已读；四大权重仅检查存在与大小，沿用既有完整ModelScope/转存校验证据，未重新读19GB权重或下载模型。

## 已冻结的后续入口

- `probe-plan.json` SHA `f16cc09a5cfc8b0dddd8555e93e39a1ce56dd8538f3b51b6ca2c99f1c83267d8`：数值门槛沿用研究报告；logits相对L2≤1%、max abs≤.25、mean KL≤1e-3；loss绝对差≤.02；cache/gradient/delta相对L2≤2%；reference L2≤1e-6时改用maxabs≤1e-6，不事后放宽；greedy8 token完全一致。LoRA fresh零B初始化的A梯度为0是正确行为，AdamW权重衰减仍可带来A delta；只要求三family有真实B更新，冻结head且全部实际梯度有限。
- `gpu_import_gate.py`：建议20秒GPU可见import-only，明确`--run-authorized`和UUID才允许入口。它不做forward、backward或benchmark，但FLA导入可能初始化GPU上下文；因此不把它伪称CPU动作。其输出必须证明两个GDN实际fused且conv reference。
- `gpu_probe.py`：要求成功的`gpu-import-gate.json`，并核冻结source manifest再启动；单GDN fwd/bwd/cache → 缓存text-only9B → 四条合成prefill/teacherforced+greedy → 128 token短LoRA一步。仅切换两GDN全局绑定，保持同模型/同输入/同形状和conv reference；native用inspect.unwrap原TF函数，fused用CPU源码审过但等待GPUgate证实的官方绑定。
- 分阶段建议160/120/220/100秒，总numeric600秒，外部timeout再给30秒TERM grace；import gate是另建议20秒，root可单独分配或从600里扣除，**此报告没有启动任何一项**。单次LoRA step耗时仅诊断，不是稳定的三重复训练吞吐结论。prefill/decode有三次热重复、交替顺序，中位数比较；冷首轮独列。
- 模型load、JSON writer、LoRA targets/family/参数盘点与因果loss复用既有`qwen35_smoke.py`（`9a15b1bb…`）及`qwen35_relation_cost.py`（`a2df013e…`），没有调用其中真实训练数据加载函数。`helper-provenance.json`保存来源。

源码、plan与manifest已同步到远端私有目录，并在CPU核全SHA；`probe-source.tar.gz`是约22KB的小备份包，local完整wheel均留仓库外。`launch-template.json`只是待root分配资源的argv，不是自动提交任务。GPU代码尚无GPU运行证据，需要root/owner在授权前增量复核源码；任何数值失败/OOM/预算耗尽都停下保存证据，不自动retry、改阈值或转正式训练。

后续最小决策：root可安排GPU可见import-only gate检验真实绑定；通过再考虑数值探针。当前没有速度、质量或0.8达标结论。
