# 全缓存初始化窄修 attempt2

原 attempt1 输入初始化被宽字符串替换误写为 `input_docs=input_docs`，5.133720秒rc1、0模型/NN/行，GPU已实际释放；完整失败与原源/plan保留。

本版只修真实JSON读取初始化，增加同入口 `--check-initialization-only` 检查模式：CUDA隐藏，执行完整相同的SHA/input schema/全部keys/head验证前缀，停在NumPy/模型/缓存之前。实际CPU通过才启唯一894秒+TERM30；两次主预算上限899.133720秒。完整源/plan及bound先push，CPU收据再checkpoint。

原408272 keys/14261 head/preparation字节同机硬链复用，不重枚举、旧shape或基准。GPU模型数学、前三批复用+795新批、compact原始分数及native raw span语义与原计划不变。当前仍没有完整缓存/fit/质量。

入口：`control.py deploy → cpu → collect-cpu → start → collect`；任何未知start先查状态，禁止重复。

实际同入口CPU检查 PASS：02:20:26–02:20:32，5.943131秒/rc0，所有输入变量/schema、408272键与14261列实际验证，CUDA隐藏，无NumPy/cache/model/NN。下一唯一894秒GPU作业。
