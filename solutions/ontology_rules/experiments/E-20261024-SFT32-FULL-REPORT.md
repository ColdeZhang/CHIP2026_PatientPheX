# 固定SFT32追加完整原文：最长输入可行性

目标是判断同一已训练SFT32能否从完整允许原文取得原局部窗口缺少的患者证据。只追加完整报告，原system/user前缀、候选/PID、adapter和greedy不变。此前base8/9及Med全文负结论保留；准确SFT32/F1同80控制未发现实跑。这是看到前序结果后的开发研究，非盲选；局部缺marker不等于可恢复TP。

CPU实际11.767秒：80提示/16文献/587CID/39PID；570passages和162给定mentions全保，去掉追加suffix恢复原80字节语义，Auto/raw/template IDs与往返均核。输入250696→670122tokens，最长14390（原idx19），1条超过旧14000。主线程明确旧14000为工程cap，现input14400/gen1600，不截输入；实际config max_position_embeddings262144，大于15990。

唯一授权A800最长idx19、240+TERM30工程probe。原adapter4f4815b6/config965644实际全SHA；恢复只读确认目标目录不存在且GPU2空卡。loader/generation复用固定旧33a源码语义：BF16 text-only/native SDPA、PEFT0.21.2/TF5.20.2、无采样、原EOS并集。新runner先持久generated IDs，再decode/raw，再严格已给CID/PID完整JSON检查；工程probe不读gold或评分。GPU前重核UUID/无进程/空间。

全文单独仅诊断，未来主臂为oldSFT32∪fulltext；两完整预测共同freeze后各唯一v2，原.660942003指标复用，1189实体及非关系字节保持。主保留条件是关系micro+macro和严格升且macro不降。完整80尚未授权，需实际最长成本/格式后统一定预算；只有精确同配置才可复用probe1条。

入口：`qwen35_sft32_full_report_readiness/probe/operations.py`，外部B=`/workspace/patientphex-runs/round2-20261023/research/qwen35-sft32-full-report-feasibility/probe-v1`。plan1ab430d2、runner062cbf90；部署/启动前先看既有intent和job，未知状态不重复start。大输入/IDs/raw/log保留外部，Git仅源和小索引。

05:23:00.013584Z唯一原driver实际启动，proc_7fce4726/PID40976，child/PGID40989，TERM05:27:00.013584，额外30秒后KILL。前置source检查点9dd171c6已push；8件实际source/input SHA与compile/fsync通过。恢复只读首检查含driver/child、尚无终点，不因观察未知而重复start。
