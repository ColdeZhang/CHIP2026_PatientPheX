# 双随机种子原生共识

当前：E-20261024-SAMPLE-001 准备中。目标检验同一SFT32模型的非思考随机输出一致性，是否能补强已有fullsupport关系。使用原 native A800 后端、同4f4815适配器与80完整全文输入，复用已完整80自然EOS的greedy G；不混FLA、不训练。

固定参数：两流20261024/25，逐行SHA256导出63bit种子；T.7/p.8/k20/min_p0/repetition1、presence0（不添加processor）、maxnew1600。各80条fresh，一次1800+30共享预算，模型仅载一次；无新probe/重试补答案。完整raw先保存；格式/非EOS/think/触顶整行弃答，不修JSON。

两个端点事前固定：primary `B∪(S1∩S2)`；diagnostic `M2of3(G,S1,S2)`。都保持B的1189实体，primary原471关系及88池外关系全部保留。160条终态后共冻，再仅各一次v2，旧分数复用、不单评seed。最高端点严格升于B .686338981可候选，primary成功另报，micro/macro无条件报告；否则关闭不扫参。仅内部F1研发，不读原dev/官网。

来源/固定计划见[plan.json](plan.json)，实验细节见[实验记录](experiments/E-20261024-SAMPLE-001.md)。外部产物：`/workspace/patientphex-runs/round2-20261023/research/nonthinking-sampled-consensus-native-20261024/`。起点main `cd4086b`，只修改本目录。源码/CPU合同/冻结plan先push后启动；UNKNOWN恢复原意图、不盲重启。

CPU真实原runtime门已20.323270秒通过：模型与adapter全部实际SHA、旧80 IDs/自然EOS、输入670122/max14390、两seed160唯一row_seed、实际Temperature/TopK/TopP/MinP处理器及CPU多项式抽样可重复。未创建模型/调用CUDA/重tokenize旧链。原native无FLA/causal_conv1d；同版本torch2.7.1/TF5.20.2/PEFT.21.2/tokenizers.23.1。完整资产/seed证据留外部。装配CPU合同另核两端点、B88池外保留、19类弃答/7类身份失败，尚无新质量。

唯一实际作业已启动：11:10:24.424427 UTC，A800 GPU2，proc_c9eb4b95-3078-4bfe-a789-0a86f4acb483 / PID10566；TERM11:40:24.424412、KILL11:40:54.424412。初次观察已3/160完整fresh，source c555f808。恢复入口外部monitor_once.py，只读原作业；绝不重启新副本。两端点的TP/FP增量诊断沿v2 association语义拆完整compound HPO，而共识票本身仍按原literal串，不改预测/门。

160/160原生fresh全量成功：两seed各80自然EOS/strict80，0cap/think/弃答，输出4707/4740 tokens；单窗口728.463375秒rc0，11:22:33.021760 UTC真实释放A800卡2（1MiB/noapps），已交sequence。506件原包347365B/1b07b483已完整回收逐SHA；160 raw/独立IDS/canonical逐行精确闭环。两个端点已共冻：primary505关系=471旧+34，diag416，两个均1189实体，primary88池外全保。当前尚未quality，准备各唯一一次v2。
