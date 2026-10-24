# Grounded 两epoch有效LoRA增量平均：准备结论

**结论：现有供给足够，值得由root决定一个有界工程/完整F1控制；目前仅CPU数学与来源准备，未合成实际adapter、未加载PEFT模型、未推理或评分。** 新变量是同Gtrain32轨迹的有效权重增量平均，不是新训练轮数或输出投票。查重范围内未发现已实际执行同一机制，PEFT库里有cat能力不代表项目用过。

## 真实资产与隔离

- 原base Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。两checkpoint共同fingerprint逐对象一致、同train32/87患者行，epoch1/global11与epoch2/global22。
- 原A100 mount `476f81cf-09e7-445b-a7f0-d0e12b12e50f`，父根 `/home/zhangyuheng/projects/CHIP2026_PatientPheX/grounded-generation-v1/formal-v1/output/`。
- `epoch-001/adapter/adapter_model.safetensors`：173181072B，SHA `b513def34303d037295cbebb3aa29f8edf5176320152bfb37e75c21f2d146549`；epoch2同大小，SHA `d52e2e6c74d3fe874940ddc9a882d3c0faa453ac362b118d1edb2782bae7c6ba`。
- 两实际adapter_config逐字节同1341B/`a9e753c4…`，r16/alpha32/scaling2；12 target families，共248模块/496 A/B张量。rank_pattern/alpha_pattern空，modules_to_save=null，bias=none，lora_bias=false，DoRA/RsLoRA/QLoRA/fan_in_fan_out均关，无target_parameters、层复制或额外token/head训练。
- 两原持久ACK `d51df102…/5f8ca5fc…` 与collector `433659a2…` 已实际fsync/reopen核四大权重。此次只读这些小原件，**没有新的远端存在性核查或大权重下载**；下一次实际消费需复核两文件。
- 祖先train F2+F3共32文献与F1的16文献交集0，原完整审`0fcffd7b…`涵盖train32词典/固定公共Sap41099。引用旧祖先证据，无新heldgold读取；quote训练锚点仍是弱标签，不因平均变成人工证据。

## 精确表示与CPU证据

原每层 Δi=2BiAi。等权平均为 `(2B1A1+2B2A2)/2=B1A1+B2A2`。令 `A′=[A1;A2]`、`B′=[B1 B2]`，新r32/alpha32/scaling1即可精确表示；只拼接，不用SVD或dense完整delta矩阵。直接平均A/B会生成交叉项，不是本方法。

已执行32个NumPy数学/schema合同，0.023474秒：FP64有效delta最大绝对误差3.55e−15，FP32 1.09e−6；Naive因子平均反例、相同/零增量、factor gauge变换不变性、shape/缺键/额外base权重/非finite/DoRA等拒绝和微型保存重开均通过。这个等式是逐线性层有效权重，不声称两个非线性LM输出或logits被平均；也不声称BF16/GPU逐位相同。

本机无Torch/PEFT，**未执行真实PeftModel加载**。已核固定PEFT0.21.2源码`32196432…`的cat分支：新rank取和、alpha=newrank，A乘weight×旧scaling后纵拼、B横拼，与本式一致。普通CPU小PEFT模块完整save/load与实际全496张量header/SHA复核是后继工程门，不需下载/加载9B大基座完成这一门。

## 原生接口、参数与成本

- 原native TF5.20.2/PEFT0.21.2/tokenizers0.23.1、A100 Torch2.6.0+cu124，无FLA；base为24 DeltaNet＋8完整注意力层，架构/词表/模板不变。全39纯输入原始文件`c6426db4…`，原native IDs压包`ee08ca3f…`与audit`51613c9c…`可复用，203588 tokens、最长10080、不截断，greedy/nonthinking/max_new1600。
- 父模型实测text基座8,953,803,264＋r16 LoRA43,278,336。拟r32为86,556,672，合计**9,040,359,936<10B**；即保守计未用vision/MTP bundle9,653,104,368再加新LoRA，也为9,739,661,040<10B。新模型尚未实例化，后继实际计数须吻合。
- 原两adapter合346.36MB，新FP32因素payload约346.23MB，CPU临时内存建议约1.2GB；常驻新adapter比r16增约173.1MB。无optimizer、训练或新base下载。精确cat提高rank，不能照搬Model soups“零额外推断/内存开销”的宣传。
- 原39完整greedy实际1103.215秒、峰21.015GB；新rank/生成长度可能改变成本，这只是proxy。建议root若实施考虑完整1500+TERM30而非断言必在旧1200内；新必要Sap queries由全输出确定，沿固定public bank，只编码未覆盖quote，不重fit/bank。
- 原`inference/run.py`有epoch2持久checkpoint、r16总参数8,997,081,600硬门。后继须使用**新的adapter合成provenance与9,040,359,936门**，保留父两ACK和训练隔离；不能把拼接adapter伪装为旧epoch2或生成虚构optimizer/checkpoint。raw-first、EOS/cap、strict quote与decoder合同不改。

## 唯一未执行候选协议

`primary = B4 ∪ (G_meanweights ∩ G_originalgreedy)`，按public-v2 literal关系原子（doc,pid,HP或精确NIL text，NO排除）；两G均为pure，不能以Bunion污染投票。保全部B4关系、原列表前缀和Task1/patient/其它对象；只加交集，不删或按患者筛选。

完整F1 B4为524关系/1189实体/39患者，Score **.701325377212172**。原Gpure282关系，其中62在B4外；62只是本控制最大新增供给上界，不是正确关系数或预计收益。全39结果和规范化完整后只冻结一个primary，再唯一1v2/old0；若零变更需byteexact继承且0v2。不单评epoch1、mean-alone或整union，不扫alpha/阈值/seed。严格胜B4并完整披露micro/mac后才由root决定后续。

同轨迹两早期checkpoint可能高度相关，平均也可能稀释已学能力；现有train loss不能证明同一低损失盆地。本方案可证伪的是固定交集可靠性增量，负结果不否定所有SWA/model soup。

## 来源与复现

- Model soups [arXiv2203.05482](https://arxiv.org/abs/2203.05482)，官方代码MIT；论文平均不同fine-tune超参模型，与本方案不同。
- SWA *Averaging Weights Leads to Wider Optima and Better Generalization* [1803.05407](https://arxiv.org/abs/1803.05407)，官方代码BSD-2-Clause；其SGD常数/周期LR与本次既有AdamW两epoch并不相同。
- PEFT0.21.2 commit `eaa35c777c7ba554a2a3639ac9c79bb690f39a57`，Apache-2.0；原cat源码和层scaling固定SHA见readiness.refs。
- 原始网页/短摘要、readiness.json、candidate-plan.json、32项CPU结果、已有证据精确refs均在本目录；无旧大包复制。源码在独立`solutions/grounded_weight_average/`。GPU/评分仍未授权启动。
