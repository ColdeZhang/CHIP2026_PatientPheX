# 一次配对质量执行

`bind_quality.py` 只绑定完整 79 fresh＋1 cache 的原始收回目录，核 terminal、全部成员 SHA、10 份推断源与适配器一致性，建立恰好七个 endpoint 引用。它不读或哈希 gold，不执行解析、模型或评分。父线程先冻结 `evaluate_adapted.py` 与模板，再运行绑定并推送实际 plan。

`run_quality_once.py --plan <bound-plan> --source <evaluate_adapted.py> --output-dir <新的quality-execution目录>` 使用独占目录和先 fsync 的 intent。子进程 `RLIMIT_CPU=(60,60)`；外层从入口计绝对 wall65 秒，再 TERM5。这里的 60 是 CPU 硬上限，额外外墙 5 秒用于普通调度/I/O，不是新的评分预算。依赖源码与输入逐 SHA 绑定，CUDA 对子进程不可见。

stdout/stderr、child 和 finally terminal 均保存在仓库外。terminal 列出实际已存在的共同 freeze、预测及 metrics，失败或超时不自动重跑、删除或覆盖目录。评分业务只调用父线程维护的 `evaluate_adapted.py`，本外壳不复制评测实现。本次只完成两份源码 compile 检查；尚未执行绑定、评分或读取 gold。
