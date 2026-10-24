# FLA 数值探针：未满足既定全模型合同，停止扩展

2026-10-24 UTC。只使用四条纯合成输入的工程诊断，没有赛事样本、标签、训练或质量评分。**保留原生 reference runtime；不据本次结果切换正式实验的 GDN kernel。**

实际 GPU 可见 import gate 用时 9.369929 秒，通过两个 GDN 函数的 FLA 绑定检查，两个 convolution 函数仍用 reference。这只证明实际分派，不证明数值或速度。

首次 numeric 在 native 单块 forward/backward 0.434534 秒后，因采样器把实际 cache 字典键当 Tensor 而退出，尚未比较 fused。最小修复只替换采样/结构比较；实际 CPU 合同核对 dict/list/tuple/Tensor/None、层/键/shape/dtype，未知类型或空状态不能假通过。原 probe/计划/模型/输入/门槛不变，修复先由检索 owner 在 `5257677ef5d648ba492fb5e721bd6b99b4630707` 推送备份。

第二次沿原绝对截止运行，未重新获得 600 秒：`proc_68ecdbc1-61f2-41dd-94e7-0c049a3ef79c`，PID 2588686；实际启动 01:01:25.098 UTC，仅剩 385.018 秒，原截止 01:07:50.116 / TERM 最晚 01:08:20.116。环境恢复后接管同一作业，没有重复启动。

| 已实际比较的范围 | 结果 |
|---|---|
| 单 GDN 块，256 token forward | 相对 L2 `.005678843`，通过 `.01` 门槛 |
| 同块 8 步 cached decode | 相对 L2 `.001225511`，通过 `.01` 门槛 |
| 同块 9 个参数梯度 | 全部有限且通过 `.02` 门槛；最大相对 L2 `.008515595` |
| 同块 prefill 加 8 步状态 | 9 次完整结构/shape/dtype 检查通过，每次 2 个状态 Tensor，原数值门槛通过 |
| 9B 模型首条 128 token 的完整 logits | 相对 L2 **`.090575123 > .01`**；max abs **`13.03125 > .25`**；mean KL `.000412400 < .001`，组合合同失败 |

首条 logits 比较失败后立即停止。其余 255/256/513 token 比较和 LoRA 梯度/一步更新没有执行。128 token 的 `infer` 内部虽调用了 teacher-forced/greedy 路径，但尚未到相应结果保存和比较行；因此不能声称其数值或 token 一致。原始大 logits 未保存，本次不能再计算居中误差或证明差异来自整体平移。

单块 fused 第一次 forward/backward **140.398562 秒**包含冷 JIT，不能拿它当稳态速度。单块三个 warm forward 中位数 native `.003092569`、fused `.006340795`；9B 的首条 cold prefill/decode 只完成各一次，既定三次交替热测尚未开始，不能给出完整模型速度结论。完整 9B 加载用时 5.717249 秒。原始 logits 的 L2/maxabs 失败、KL 通过，两者如实分列；这不是赛事质量下降结论，也不能直接推断根因。

01:04:40.779 首次恢复检查已发现进程退出且目标卡为空；01:04:59.035 再次实际核对 GPU1 为 0 MiB / 0%，没有该 PID/该卡 compute app，managed API `running=false`。API没有返回独立 exit code，不虚构 `rc=1`；完整 stderr 指向 `AssertionError: synthetic-0/logits`。从第二次启动到首次确认退出的上界为 195.681 秒，非原生精确总耗时。

全部实际小产物保存于 `attempt2/`；`attempt2/terminal.json` 为机读终点，`gpu-result-backup-filelist.json` 为 owner 入库清单。完整正常日志留仓库外，原始失败与成功 CPU 检查原位保留。没有第三次重试、放宽门槛、正式模型迁移或官方上传。
