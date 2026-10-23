# Qwen3.5-9B最小受控对照提案

15文件实际SHA已完成，私有runtime CPU导入、真实token预算和接口准备通过，尚无新GPU结果。首轮启动前预留卡已被他人占用，等待重新调度。主线程已授权短接口smoke→单条最长输入推断→最长训练样本单步成本，正式质量作业待该反馈后安排。目标是提高指定患者关系判定，不重新生成自由HPO编号。

## 固定输入与比较

复用CRF已独审的nested OOF204训练行（48篇、内3折train32候选；无gold补候选），SHA`eb133c3107c1aada2e156892ba430e5b5860315c058d6c7dd5387fe45b065f77`。保持原58个fold0推断messages字节，SHA`d387f5b8e7b714e81619f5972a6385925629bf7b2e4be3ca55936c82a1077ff0`。模型只输出可逆候选C码对应的患者集合，保留原患者标记、候选顺序、上下文、空目标。来源为外存`sequence_crf/relation-sft-fold0/`和`sequence_crf/qwen-pair-fold0/prompts.jsonl`。

Task1实体固定原CRF+Sap947个；后处理使用sequence原`merge_pair_rag.py` SHA`817fdc4494e7ed349085fe97c5744409d66d0f1de476089ac8c5696a84a6b4a1`、原LR fallback，不新造患者、候选或阈值。既有同输入8B base Score .623068497、8B2ep adapter .674333431、规则 .711353522。新基座在A100/torch2.6cu124/TF5.20，旧8B在A800/torch2.7.1cu126/TF4.51；这些是同任务对照，硬件/runtime差异如实披露，不宣称只有模型规模单变量。

## 最小顺序与成功证据

1. 已授权接口smoke：实际官方tokenizer从自己的train48抽取数据选择最短非空答案且完整≤512token样本；无样本则失败，不截断。text-only BF16/SDPA/no-packing、r8 alpha16 dropout0，targets `in_proj_qkv,q_proj,v_proj`覆盖DeltaNet；1step有限loss/非零梯度/adapter B变化、保存重载实际参数一致、16token greedy。只验证接口，smokeadapter不作正式起点或质量证据。实际文本参数应8,953,803,264，连adapter总量<10B，缺失/错配文本权重直接失败。
2. 正式质量候选（尚未启动）：原58输入、batch1/maxinput14000/maxnew1600/greedy/thinking=false。先按实际新tokenizer选择唯一最长6040token输入做格式/显存/耗时测试，不读gold；若可运行，完整58重新独立运行。所有失败保留并按原merge回退，严格记录有效decision/截断/格式错误和同实体关系P/R/F1/Score。
3. 前两步通过后已授权一次独立fresh base最长5787token训练样本单步成本：完整r16/alpha32/dropout.05、12类projection、248模块共43,278,336 LoRA参数，连文本基座8,997,081,600<10B。BF16/SDPA/no-packing/nonreentrant gradient checkpointing，仅单例单步，核三类梯度/峰值/时长，smoke权重不作为训练起点。是否正式训练204×2epoch由成本及root后续决定，不预先启动。新模型LoRA投影需按新架构明确匹配，不能声称旧7targets覆盖DeltaNet；训练完整无截断、assistant-only、新base初始化，绝不沿用8B或smokeadapter。原dev不参与选型，内折若无收益不扩展参数扫描。

## 真实token与预算

新tokenizer204训练总660,330tokens/epoch、max5787（旧5799），其中prefix648,409、监督11,921；完整prefix一致、EOS存在，无截断。58推断195,355tokens、max6040（旧6042）。原同任务8B A8002ep为599.86秒/峰29.48GB，base58为164.94秒；9B含纯torch DeltaNet fallback，不能照搬这一耗时或显存，须测短smoke和最长输入后再给实测预算。源15文件19.329GB、A100项目初始可用93.33GB，下载8并发/64MiB Range，约8–9MB/s只是当前观察值。

Qwen3.5固定text config EOS=248044(`<|endoftext|>`)，chat template EOS=248046(`<|im_end|>`)，无独立generation_config。生成显式使用两者作为结束符，训练仍监督官方模板原字节。与8B的im_end/endoftext双停止语义一致；必须写入实际manifest，不把重复到触顶当作纯模型缺陷。

新入口`qwen35_runner.py`始终核完整源SHA和text参数、每条完整消息预算、严格恢复指纹/覆盖；不改旧8B入口。5项CPU恢复/患者身份/训练答案隔离检查通过，真实GPU加载与生成尚待接口检查。
