# Full64 选监督 CE 显存恢复

原完整训练已 OOM、无完整 epoch，不能恢复旧 optimizer/RNG。此目录是独立 fresh 恢复；保留完整 BF16 head，只在转 FP32/native CE 前选 causal-shift 后有监督位置，per-row mean 和实际 group 分母保持。旧失败调用在 backward，未定位具体 CUDA 算子，5.71GiB 仅与完整 FP32 张量形状相近。

- `selected_loss.py`：与 actual Torch CPU 288 例合同相同函数 AST；原生 fixed CE，不改全局映射，只赋实际模型实例 loss property。
- `probe.py`：硬 240+30，原第一组8条一次非零LR更新产生 Adam，随后原第12组前三条累计，再最长6166 backward。逐阶段记录内存/身份，无 del/gc/empty_cache，不改 allocator。一个真实更新、12次 forward/backward，状态全部丢弃，无质量。
- `formal/`：**已获root统一授权，05:13:13Z唯一正式job运行中**。fresh265×2/68calls/4warmup，原4500+30含两个完整持久state，完整epoch2才原dev64另900+30。唯一新质量为a179全预测并新患者关系，全部raw/fullpred冻结后一次v2，原政策不变。不得从probe权重续训。

实际入口与终点只看路线 WORKLOG、外部`research/patient-first-full64-memory-recovery/`及小证据索引。原intent/job中的Git metadata一位错误有独立纠正，原source/plan SHA真实无误；不要重启既有job。
