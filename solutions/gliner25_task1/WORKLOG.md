# GLiNER2.5 Task1 工作台账

2026-10-24 UTC，分支 explore/gliner25-task1；仅维护 solutions/gliner25_task1。

- **当前：固定配方完整负结案。** 强 Task1 基线 .728697536469 → 全部接受实体并集 **.724624277992（−.004073258477）**；新增104=23TP/81FP。new-only诊断 .665067495215。主门失败，不迁F0/原dev，不调cut，不新增训练。原dev最高仍 .754361398199。
- **终态入口：** [final.json](evidence/final.json)→外存 terminal.json，一处绑定源、3阶段完整回收、原job终态、2份预测/格式/metrics/共同freeze。详细过程与解释见 [实验](experiments/E-20261024-GLINER25-TASK1-001.md)。集中独立复核尚待root安排，未冒称已审。
- **实际执行：** fresh32篇/320窗/10epochs/1600更新342.260秒；16篇/39患者/164完整F1窗23.942秒，1466selected全合法；去重1287跨度/764quotes，398必要Sap12.040秒，接受886原子。训练、推断、规范化均rc0。
- **资源已释放：** .100GPU0 11:58:02.427201 UTC 0MiB/noapps；.3GPU0 12:03:21.650951 UTC 4MiB/noapps，已交Qwen owner。当前无GPU请求，无在途模型作业，不重复启动旧proc。
- **质量范围：** 两份完整16篇/39患者预测共冻9b3be589后恰好2次新公共v2(all/text)，旧0。主1417实体/诊断886；原关系和其他对象精确保留、Task2分数不变。主pred1829b9c9，诊断fc68d825。它们是内部F1研究，不是原dev/官网成绩。
- **完整原件：** 正式642小件/6外部大件、推断508件、规范化17件均全SHA回收；原始候选/拒绝和分数保留。epoch5与final的模型/optimizer/scheduler/scaler/RNG读回一致，未声称原生自动完整resume。[训练/推断入口](evidence/formal-inference-complete.json)。
- **工程边界：** .2随机query增强未关；每轮全部320窗、19200raw loss分项有限，606实际schema保存。38个词内正annotation不适配固定splitter，提前审计且未扩边界；NO仅负上下文，未伪称全部pair显式负训练。前两次probe工程故障及纯CPU元数据恢复都保留。
- **恢复目录：** R=/workspace/patientphex-runs/round2-20261023，本线R/research/gliner25-task1-v1。唯一后继是集中终审和小产物备份，不重评既有端点。
- **最近已核远端：** 5adbb4ff；本次完整结案与小备份摘要随后提交推送。

下一步：将统一终态/源交root作一次集中复核，保留23个独有正确实体与全部score供另立假设；当前配方已关闭，不将模型家族整体判死。
