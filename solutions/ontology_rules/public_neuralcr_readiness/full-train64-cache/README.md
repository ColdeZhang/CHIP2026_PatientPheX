# NCR train64 纯输入完整缓存

目标：完成冻结的 408272 个 token keys，解除新 HGB 实体链的公共 NCR 特征供给不确定性。当前仅准备，尚无全缓存、fit 或质量终点。

- 固定 64 文献/2301 段/683438 原生 1–7 词候选；远端以原 AST 从四字段 input-only 重建，必须与既有 16445155 B 键文件 SHA 完全相同，所有坐标与 token 代表串 roundtrip，随后冻结 bound-plan。
- 原完整 10 ensemble/491068890 神经参数，原生 FP32、TF32 关闭，60 数组实际数值 exact；NumPy 1.23.5 保留原 float32 `>.8` 标量语义。旧 3×512 按原矩阵/键/头序和 score bits 核后复用；795 新批/406736 新键，合计 798 批/408272 键，末批 208。
- 一个 900 秒主进程+TERM30，fresh UUID/空卡后启动；保存实际逐批进度、完整 SHA、退出及释放。所有中间数组/日志在仓库外。固化的 223–337 秒成本仅先前 prefix 探针外推，未当成本次实测。
- 紧凑 NPY 每行原始 uint16 列+float32 分数（6 B），完整 payload2449632 B；键序/head列序 SHA 绑定，不传回新全维概率矩阵。
- 同一原 annotate_text AST 从缓存离线解码原 HPO spans，逐span绑定原文坐标、cache行、原HPO及score uint32；无额外NN。固定512批缓存不宣称与任意原逐passage不同batch shape的NN输出bitwise等价；四合成parity仅限实际样本。当前HPO映射随后单独做，raw spans不能冒称赛事最终候选。
- 任何失败原样保留，partial不可冒充完整；unknown start先核状态不可重复启动。无 gold/fit/质量/原dev/官方。新实体链须去除B候选+6列、fresh vectorizer/HGB；旧97列关系支持器仍保留，不能当全系统零依赖删除。

入口 `control.py deploy → prepare-start → prepare-collect`；实际准备通过后冻结并push `bound-plan.json`，`publish-bound → start → collect`。所有阶段小源/plan和关键收据及时提交推送；完整资产与运行目录 `/workspace/patientphex-runs/round2-20261023/research/public-neuralcr-readiness/full-train64-cache`；远端同名相对目录。
