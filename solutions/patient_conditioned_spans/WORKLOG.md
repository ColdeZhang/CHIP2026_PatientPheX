# 患者条件跨度路线台账

当前：原8轮控制完整有效负结案并终审；已获批独立 E-20261024-PCS-002-DURATION24，fresh24轮/768步时长对照已唯一运行。

目标是改善患者归属与候选池外漏概念，检验整文共享编码、患者条件跨度与概念级 max-MIL；不重复已负的公共候选对 DocRE 或无患者条件 GP。

- 依据：[实验方案](experiments/E-20261024-PCS-001.md)、[注意事项](NOTES.md)、[证据索引](evidence/cpu-ready.json)。
- train32 F2+F3 / pure F1：87 / 39 患者，所有文献、锚点、全文窗口均保留。
- CPU-v2：813 个可监督正袋、1 个 masked 正袋；3 个不精确词片正跨度的覆盖候选不作为 NER 负例。完整数目与口径见实验。
- 下一步：同源同图同loss，仅fresh24轮/768步/warmup76，复用原CPU与两图probe，900+30，完整两端点共冻后2v2。工程原120窗口实际15.558s，峰allocated3.755GB，参数109187713，两个整文形状全部真实更新通过，12:40:06.810949实际release。CPU原start UNKNOWN已恢复同PID完整终态，无重启。
- 禁止依赖 held gold 调 cut；旧8轮已完整闭环，24轮尚未形成新质量。原 sampled 成功记录及所有旧路线保持原样。

事前修订：86b37baf 的 balanced-BCE 准备保留且未执行任何 GPU/质量。为保持本次关键变量为患者条件和 latent bag，正式 NER 改用原 GP 的 multilabel categorical loss；完整几何不变，新增高分稀有负例影响/有限梯度 CPU 合同。

公开Sap41099库已.3→.100同SHA实际复制/fsync重开，1.5245秒；使用原policy .90 cosine/.03 distinct margin及全部字符串保护，无重编码。证据 [工程结果](evidence/probe-complete.json)。

正式790fb2be：12:44:47.459117 UTC / proc_72ec2375-4ed8-4467-b534-21d86c477aeb / driver2739073，native2739081，绝对截止12:59:47.459117+TERM30。D/formal-v1/started.json；只恢复原任务。原38.914秒观察epoch4 step97/256；规范化与训练共享此时钟。

正式完整107.900268秒rc0，12:46:35.358142释放；256更新、全16/39推断及137规范化关系。88件原终包866369B/5b355298全SHA收回。首次装配错误地从预测取patient字段，首个预测/gold/v2前退出；另assembly-interface-recovery保留失败与原源，机械改从pure input核患者，不重模型/规范化，进入assembled-v2两端点。


最终：强B .686338981272 → 保护B新增主端点 .678196119397（−.008142862），micro .716763 / macro .581083；47新增6TP41FP。仅new关系诊断 .461223490945。全部1189实体/471旧关系保留，2新v2、旧0。详见[完整终点](evidence/terminal.json)。

证据：原88内容包866369B/5b355298完整回收；48质量内容包121117B/4b45c790已远端文件fsync、重开全成员SHA及完整readback。末轮部署checkpoint436837342B/f556e751留.100，文件fsync与SHA实核，无optimizer/RNG或数值reload。

只读后继诊断：两训练分项在第8轮仍下降；813正/2381负/1masked，256步。仅建议事前固定fresh24轮/768步检验优化充分性，估263秒；尚无新GPU授权，不自动重训/扫cut。当前原dev最佳ee39保持不变。

后继root明确授权duration24：见 [独立计划](duration24/README.md)。原8轮所有源、模型和负质量均保留，新checkpoint只部署，不是full-resume。

24轮实际：12:54:49.207975 UTC / proc_73c003dc-6bee-4ad3-a779-d1a8b98f4635 / driver2739779 / native2739790，绝对截止13:09:49.207975+TERM30。源d487f2eb/计划55565bc4，所有原payload与model字节复用。当前只恢复此原作业，不重启。
