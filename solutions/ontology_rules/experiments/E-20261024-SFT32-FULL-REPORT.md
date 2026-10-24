# 固定SFT32追加完整原文：F1完整80对照

目标是判断同一已训练SFT32能否从完整允许原文取得原局部窗口缺少的患者证据。只追加完整报告，原system/user前缀、候选/PID、adapter和greedy不变。此前base8/9及Med全文负结论保留；准确SFT32/F1同80控制未发现实跑。这是看到前序结果后的开发研究，非盲选；局部缺marker不等于可恢复TP。

CPU实际11.767秒：80提示/16文献/587CID/39PID；570passages和162给定mentions全保，去掉追加suffix恢复原80字节语义，Auto/raw/template IDs与往返均核。输入250696→670122tokens，最长14390（原idx19），1条超过旧14000。主线程明确旧14000为工程cap，现input14400/gen1600，不截输入；实际config max_position_embeddings262144，大于15990。

唯一授权A800最长idx19、240+TERM30工程probe。原adapter4f4815b6/config965644实际全SHA；恢复只读确认目标目录不存在且GPU2空卡。loader/generation复用固定旧33a源码语义：BF16 text-only/native SDPA、PEFT0.21.2/TF5.20.2、无采样、原EOS并集。新runner先持久generated IDs，再decode/raw，再严格已给CID/PID完整JSON检查；工程probe不读gold或评分。GPU前重核UUID/无进程/空间。

全文单独仅诊断，未来主臂为oldSFT32∪fulltext；两完整预测共同freeze后各唯一v2，原.660942003指标复用，1189实体及非关系字节保持。主保留条件是关系micro+macro和严格升且macro不降。当时完整80未授权；后续依据实际最长结果获得下文1200+30统一预算，精确同配置复用probe1条。

入口：`qwen35_sft32_full_report_readiness/probe/operations.py`，外部B=`/workspace/patientphex-runs/round2-20261023/research/qwen35-sft32-full-report-feasibility/probe-v1`。plan1ab430d2、runner062cbf90；部署/启动前先看既有intent和job，未知状态不重复start。大输入/IDs/raw/log保留外部，Git仅源和小索引。

05:23:00.013584Z唯一原driver实际启动，proc_7fce4726/PID40976，child/PGID40989，TERM05:27:00.013584，额外30秒后KILL。前置source检查点9dd171c6已push；8件实际source/input SHA与compile/fsync通过。恢复只读首检查含driver/child、尚无终点，不因观察未知而重复start。


## 最长真实终点及完整80授权

13件终点已完整回收、逐SHA：manifest ed9f4e94。14390输入→70输出，native6.547998秒、wrapper33.770421秒rc0，load5.186099秒；完整供给CID/PID严格JSON与自然EOS、无cap。峰allocated22,233,742,848/reserved23,502,782,464字节；05:23:33.776673实际GPU2无进程。raw-first保证从generate返回后开始，不保证生成中OOM/TERM仍能回收部分IDs。

据此root统一授权一次完整80、1200+TERM30，idx19精确缓存1＋79fresh，同input14400/gen1600及原模型/adapter/greedy/math flags；不自动扩时或再生成。旧80 282.502秒×输入token2.673只是约755秒成本代理，非保证。新source9bb2b322/plan73ebed7c，按原长度/row_id排序，逐行IDs→raw持久后按probe严格供给CID/PID完整JSON校验；EOS/cap/格式失败整行弃答、无补负答案。完整两个端点共同freeze后才各唯一v2，1189实体与全部非关系literal bytes保持，旧分复用；不自动迁F0/dev。当前source准备、无新质量。

完整后继CPU评测源已冻结：8件56455字节，manifest84346afc，evaluator ef28ec27 / template3b170a4e / binder1251ce92。合成自检约0.01秒PASS，模板保持disabled，只有完整80、rc0、卡释放和240份逐行证据核验后才绑定；共同freeze后首次gold，固定新2次v2/旧0。静态增量62f1a2c1无阻断，实际评测尚未执行。

唯一完整80实际启动：proc_cce6e47e-4136-4340-ba79-f90fa5986319 / driver53156 / child与PGID53166；05:32:47.863925 UTC开始，TERM05:52:47.863925、另30秒后KILL。部署13件7片全部实际SHA/compile/fsync通过，启动前af32f19已push。首次只读检查原driver/child存在、尚无终态；不重复启动。

## 完整质量结果

05:39:04.189577原完整80结束，376.475652秒rc0；实际79fresh、1精确probe缓存，80行全部严格完整JSON与自然EOS、0弃答。fresh生成347.081604秒，load5.827538秒；峰allocated22,047,325,184字节、reserved74,941,726,720字节，两者不能混为实际活跃张量；05:39:04.322610卡已无进程。253终点成员/92697字节zip7740a0bc回收，240原行IDs/raw/canonical逐SHA核。

binder plan3e3cea20，两个完整预测共冻f0ccf03c后，CPU0.334秒固定新2次v2、旧0。仅已知F1开发集，不称独立盲选或原dev成绩：

| 端点 | Score | 关系micro | 关系macro | TP/FP/FN |
|---|---:|---:|---:|---:|
| 原SFT32缓存 | .660942003 | .686600221 | .542228853 | 310/73/210 |
| 诊断全文单独 | .676546355 | .695278970 | .595967513 | 324/88/196 |
| 事前主臂旧∪全文 | .669554799 | .694092827 | .569187435 | 329/99/191 |

主臂满足原micro+macro严格升且macro不降，retain=true，Score增加.008612797。它新增45关系评分单元，其中19TP/26FP；全文单独比旧净+14TP/+15FP。用并集关系可导出全文独有19TP/26FP、旧独有5TP/11FP，故单独结果更高仍须按事前诊断身份报告，不能事后改主选择标准。全部1189实体及每行顶层association以外原UTF8字节保持，mention/doc指标完全不变。

这支持在本已知F1上全文证据有用，未证明所有缺患者marker的旧FN可恢复，也未证明原dev改善或0.8达标。下一步由root统筹，不自动迁F0/dev/官方；终点独审和一次小型持久备份完成后更新结案指针。原负R1与其他全文控制不被本结果改写。
