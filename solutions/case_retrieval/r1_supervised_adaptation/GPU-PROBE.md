# R1真实工程探针：数值断言停止

2026-10-24唯一A30 GPU1探针在最短570-token样本的原生/selected CE真实LoRA梯度比较处停止：`torch.allclose(rtol=1e-4, atol=1e-6)`未通过。程序29.012秒、外层29.99997秒/rc1，显存peak allocated19,495,760,896B。**0次Adam更新，尚未执行12步累积／最长5519压力测试；没有质量分数。**完整[结果指针](gpu-probe-result.json)。

已通过的部分是：9件模型资产实际SHA、目标卡空闲门、BF16实际基座和参数finite、原95fb派生配置及Yarn数学对照、252目标模块/43,646,976 LoRA参数、全基座及head/embedding冻结。原124编码与源沿CPU合同，不增加新gold。2次最短前后向已执行，差异断言发生在它们之后。

本次诊断实现有记录缺口：先逐tensor断言，再准备写差值，导致具体loss/梯度偏差没有持久保存。不能从该断言推断原因，不能称selected CE数学目标不等价，也不能称训练显存或质量失败；CPU合成loss/gradient一致证据仍只代表其覆盖范围。没有放宽阈值、热改或自动重启。

11份远端小文件已回收逐大小/SHA校验，原始stderr与失败均保留仓库外。实际只读检查确认GPU1恢复0MiB/free24164MiB，无该卡compute进程；GPU0原他人任务未触碰。探针无adapter/checkpoint，状态已丢弃。

下一步建议仅补数值观测：同最短样本保存每组loss、逐tensor和全局梯度绝对/相对L2差异，加原生→原生同RNG重复作为可重复性对照，再判断实际差异；该后继现另获root授权180+TERM30，独立源`numerical_diagnostic.py`/`numerical-plan.json`先freeze/push再执行；本次原失败不改写。完整训练预算仍缺真实Adam/最长累积实测。

原探针统一终审`615e4193…`核实失败范围与实际释放；不重复CPU或旧证据。
