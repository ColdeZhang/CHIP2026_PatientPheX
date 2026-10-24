# 独立 FLA A 数值诊断

本目录只含 A：128-token 裸基座全部 logits，以及真实14390-token old4f48 完整上下文固定32位置＋8步共同 teacher-forced 历史。各组 N1/F1/N2/F2，同输入/相同起始 RNG/不更新权重；convolution native，只在独立子进程切两个 GDN 绑定。没有任务答案、金标、质量、训练、旧数值门重算或自动 B/C。原失败仍引用历史证据，所有新统计都没有通过/迁移阈值。

固定32位置 `floor(i*14389/31)`，head前索引由原 modeling 接口执行；引用已独审 e35dba61 头部合同，仅检查本次实际索引，不重复该数学合同。包保存原 tensor dtype，先 fsync/SHA/reopen逐字节，再写finite统计并判定；NaN原件保留。总raw tensor上限1GiB，长输入不产生14GB全logits。CPU阶段已实际验证常量平移、32位置和NaN保存先后，见 cpu-result-pointer.json；没有导入torch/FLA/Triton/Transformers。

父线程先部署 CPU 的 result.json/fixed-inputs.json，运行 bind_cpu.py --template template.json --cpu-dir <local-cpu-v1> --remote-cpu-dir <remote-cpu-v1> --output <ready-plan.json>，再加自身 launcher_sha256 并提交。CPU与template原件不改。源中不含远端调用或自动启动器；launch_owned.py 单独归父线程维护，不进本源manifest以避免互相散列。

GPU CLI：probe_A.py --plan <ready-plan> --output <exclusive-actual> --absolute-deadline-monotonic <parent-start+600> --run-authorized。父线程600秒绝对预算包含GPU门、完整模型SHA、load/JIT/I/O，子进程在deadline−30即570秒前停止新增工作，TERM宽限30。失败保存原件，不重启、不延长。source环境严格复用已有private FLA wheel；新检查会逐.py核安装包与旧wheel完全相同，不能静默native fallback。

输出：actual根intent、绑定/资产、base-loading、raw-budget-progress、progress、result；两组下N1/F1/N2/F2各identity、prefill.pt及先raw后observation sidecar、cache结构、complete；长组另8个decode.pt。四种对比是native_self_repeat、fused_self_repeat、cross_first、cross_repeat。按位置输出raw/中心化误差、双向KL/probability L1/logprob、top1/margin；缓存只保完整结构，不宣称缓存数值相等。首轮和一个重复的同步耗时逐次记录，不冒充三次稳态吞吐。
