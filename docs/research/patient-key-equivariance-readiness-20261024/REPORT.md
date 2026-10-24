# 患者键置换：当前接口阻断，不启动控制

2026-10-24，只读现有代码与 train32 汇总。**未发现独立模型内部患者 code；按本次边界不立项直接 patient-ID 循环重命名。** 未构建改写器、加载 tokenizer/模型、fit、评分，也未打开 F1/dev gold。没有运行新的数据普查。

**键的真实含义。** 冻结 renderer `sft-meta1-fusion/prepare_pair_rag.py` 直接把 `doc.patient[].patient_id` 用于 `patient_ids`、全局 anchor 行首、`[PATIENT ID: 原文]` marker。patient-first 的 `transform()` 仅修改 system 前两行并转置训练答案，整个 user message、PID 值及患者顺序原留；JSON target 的 key 同样是原 PID，没有 `patient_code → original_id` 映射。官方缓存 README 第154行说明：“文章出现具体编号则 O+文中的编号，其他为 P 开头按出现顺序编号”。`patient_id_anchors.py::clinical_identity` 也据此只将 O 前缀解析为临床编号，拒绝把 P 序号当作临床身份。不能统一把所有 PID 当任意匿名符号。

**α=.5 的 opaque key 不等于输入代码。** `prepare_weights.py::patient_key` 对 `(pmc_id, patient_id)` 加盐哈希，仅在 loss metadata 中标识同一患者、核患者内 chunk 次数；renderer 和 target 不会看到该哈希。现成系数报告只证明 32篇/87患者/124行、346个患者块和4694个监督 token 的权重结构；患者 key 的900个纯 key token及混合 token 不是“已证明存在编号偏置”的证据。它没有按 O/P 类型或患者顺序测因果差异，本次不新增这种标签诊断。

**为何不直接改字符串。** 循环交换 `OII.1`/`OII.2` 而保留原文病例编号，会让 marker/答案键所暗示的临床编号与原文患者身份冲突，即使 JSON 矩阵能够机械反变换，也不能称语义不变的内部代码置换。对原文病人编号做替换更不符合任务边界。当前 renderer 还在含 PID 的 anchor 整行上按字符预算截断、对共址 marker 的 ID 排序，并对已标记 context 去重；重命名后重新渲染可能额外改变可见证据，不能只校 target 逆映射就宣称输入一致。

**历史与机制区别。** 三条活跃路线的有限 Python/Markdown 检索未发现 patient-ID permutation/renaming 实跑。候选 reverse 明确保留患者列表及顺序，只更改 CID/HPO 顺序；patient-first 转置保留全部 PID/user 文本；coref marker 扩充的是患者 mention。α=.5 改同一真实患者在各 chunk/列表值的损失权重，亦不改变患者代码或输出顺序。它们均不能当作患者重命名控制。患者顺序本身是否造成偏置仍未知，此阻断不等于证明模型具有顺序不变性。

**最小否定证据与去向。** 源码直接复用原 PID、官方 O/P 语义、缺少独立 code 映射，已足以否定“在当前接口上只做任意内部键置换”的前提，故不自动实施 CPU/NN 控制。将来若专门引入独立 code 层，须把原临床 ID 作为不变身份属性保留，并建立结构化 marker/target/解析器的双向合同与证据字节检查；那属于新的 prompt 表示设计，不能冒称当前控制的无语义变换。本次不扩展该设计，不干扰当前训练。

全部证据的路径、大小及 SHA256 见同目录 `sources.json`；没有修改任何路线源文件或共享文档。
