# 患者优先 full64：CPU 和最长样本探针

目标是检验两折均有小幅互补的患者优先关联能否迁到原 train64/dev16。三方并集已通过固定双折门槛，但 F1 micro 下降，F0 仍低于 rules；不能预期必然提高原 dev。

## 当前边界

只授权 CPU 数据与单次 A100 最长行更新，240 秒加 TERM30；正式训练尚未授权。后续若运行，用 fresh 两轮 265 行/68 次 optimizer calls/4 warmup，拟 4500+30 含两完整持久状态；dev 原64输入拟另900+30。最终必须在当前 a179 全预测上保留全部旧实体、关系、public-new支持，仅检验新增患者关联，不重新计算或删除旧候选。当前未读 devgold、未评分、未官方上传。

## 实际 CPU 数据

既有冻结265条监督逐行转置，4条48→16候选来源链均排原dev16。旧 native 编码逐行及全量哈希一致，5328对（1229正/4099负）双向恢复原答案精确。新总860405 token、9818监督；唯一最长3563609:relations:0为6166/6044前缀/122监督，小于旧6400上限，零截断。dev64/16篇/1006候选对仅输入，旧IDs逐行精确，新最长5198。纯CPU实际7.988秒，正式训练环境 native265 检查待执行。

## 最小 GPU 验证

复用已审 F0 数值主体，fresh公共9B+r16 LoRA、BF16/nativeGDN/SDPA、非重入checkpoint、原生完整logits CE、单次非零lr1e-4/AdamW .01/clip1。只adapter进optimizer、base冻结；检查三族B真更新，允许零初始化B造成首步A梯度为0。探针不存权重、不作为正式起点，不修改allocator/输入/loss。首次OOM、非有限或超时即停，保留终态，不自行重试。

新源 `patient_first_full64/`；原数值段逐字比对及6输入绑定见 evidence/patient-first-full64。外部入口 `/workspace/patientphex-runs/round2-20261023/research/patient-first-full64-probe/`，原CPU完整交接为同级 `patient-first-full64-cpu/handoff-manifest.json`。小摘要进Git，全数据/日志/后续权重外留。

## 实际 A100 native265 入口

隐藏 CUDA 的实际 TF5.20.2/PEFT.21.2/Torch2.6 CPU预检通过；两个 GDN binding 来自原 native 源 b4b17f，FLA不存在，allocator变量仍null。265条全部编码/labels及聚合SHA与CPU准备精确，实际 8.432秒；最长selected完整文件3beacc4e与原CPU精确。receipt8137d850，未读dev目标/单独dev输入文件或权重，CUDA未初始化。16件源码/数据部署与新receipt绑定后，将执行唯一240+30探针；完整训练仍待结果。

04:33:22.332Z唯一GPU探针已启动 `proc_281ec4d2-bef3-490c-bdd7-4d60f43dccae` / PID113890，fresh GPU2核空、源与实际计划3481172已推送；近似TERM04:37:22/KILL04:37:52，以remote driver实际截止为准。只读回收 `patient_first_full64/operations.py collect --directory <外部probe目录>`，未知状态不得重复start。正式trainer仅绑actual native与既有full64配置，未生成正式可执行plan/未launch。

## 最长行实际终点

唯一探针81.873秒rc0，native80.700秒，前向/反向/真实更新15.167秒。loss .252976、梯度范数3.105489均有限；DeltaNet/full-attention/MLP三族B真更新，248个B梯度非零，A首零符合初始B全零。base无梯度/仅adapter优化，FP32优化器状态。allocated峰38,183,758,336 B，reserved峰39,082,524,672 B，结束测得free2,665,807,872 B；free不是峰值剩余，也不证明accum8和多形状全程。未变更allocator/attention/loss/精度/输入，未重试，未保存权重。

真实04:34:44.426409释放GPU2，0/40960MiB且无进程。8小件逐SHA回收，step2d94cb44、wrapper700a210d；完整探针清单见evidence。第一次回收读取先见complete、未见wrapper；随后原文件回收证实实际成功终点，不重启作业。source与job已d313452推送。

正式计划参考同A100 F0两轮2279.145秒，新token比1.309922，线性参考2985.501秒；拟4500+TERM30包括每epoch完整状态fsync/祖先链持久/原字SHA回读。原dev64另900+30，原max6400不截断，fresh两轮265条/68calls/4warmup。正式trainer已绑定actualnative但尚未运行，full64正式是否启动由root根据实测统一决定。纯推断草稿保留先原始IDs后raw再有限codec，三项epoch2真实收据仍pending。
