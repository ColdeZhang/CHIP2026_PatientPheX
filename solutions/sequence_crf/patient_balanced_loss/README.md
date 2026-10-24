## CPU完整终点与唯一后继

原CPU已完整回收8件，317观察allPASS，worker5.207856秒/driver5.690608秒；collection9f0bf341、resultcd96f25b、observationsb3a8a9d7。独审8765aeb8已PASS，包含完整系数桥接、12新输入编码与hook/driver静态增量；不重执行CPU。见[唯一终态](CPU-terminal-summary.json)。后续仅已授权240+TERM30工程probe，源/真实输入绑定在[probe-plan.json](probe-plan.json)，H5 proc431edce8释放后fresh核；probe未启动，formal和质量尚未授权。

## 新授权与待运行探针

root已授权在原CPU全部原件核验后、H5正式600控制实际释放GPU2之后，沿原A100环境唯一240+TERM30工程probe。8微批非零Adam→3累积→最长5556，共12次反向，状态弃掉；不是formal训练，没有评分。`probe_prepare.py`只重新编码这12行并对原native四SHA，既有全124位置/Fraction合同不重跑；probe源检查点先推送。

`probe.py`保留全BF16 LM head后selected FP32 CE，通过`row_loss_scope.py`为每行捕获不可变metadata与真实g，在forward至backward重算结束保持绑定并try/finally恢复；实际生命周期、梯度/optimizer state有限性、三族B变化、base冻结和显存必须由一次真实probe证明。`probe_driver.py`绝对240+30且fresh UUID/process/空间gate，非零Adam只是工程诊断，正式原warmup2不改。未来质量预定在[quality-design.json](quality-design.json)：主N∪R∪newP替旧P；诊断newP-alone；保1189实体/39患者/非关系，双完整pred共冻→各1v2。旧N/R/P .663841820660及强全文 .676546354662594均列，不把低于强参考的分数称全局best；formal/质量预算尚未给。

## 实际进度（2026-10-24）

原生位置与权重准备已唯一执行成功，2.716751秒，manifest `c713111b…`；全部原124行IDs/labels/audit exact、两轮每患者新份额1/87及混合1/174精确成立，跨两个患者值的token为0，prepared空列表106处保留。见 [小型实际清单](preparation-actual.json)。

后继 [CPU数值计划](cpu-contract-plan.json) 固定原A100 Torch2.6.0+cu124、CUDA隐藏/线程1、60+TERM5；未执行。只用vocab11合成labels/logits，核原native CE、独立显式患者均值/解析梯度、FP32/BF16、NaN/Inf忽略区、原124行实际系数对Fraction桥接。α0仅合同锚点，不是新训练臂；不加载模型、未hook trainer、不fit或质量评分。实际源码/计划先推送，再按 `run_cpu_remote.py pack/deploy/start/collect` 唯一执行；任何未知结果先库存，不自动重试。

以下保留原准备协议；其中“未执行”描述对应事前状态，以本节实际状态为准。

# 固定 α=.5 患者等权混合目标：CPU 准备

这里只准备原 patient-first train32 的 124 行目标位置及系数。父线程推送源码和计划后再唯一执行 `prepare_weights.py --plan preparation-plan.json`；此阶段不授权 GPU、模型加载或训练。`mixed_loss.py` 与梯度 CPU 合同由父线程独立负责。

训练文件 `14af86cf…`、原 native token audit `36f6faae…`、tokenizer `5f9e4d49…`、template `a4aee8af…`、provenance `356109e9…` 和 trainer `a3ec17f5…` 全部冻结。只用 AST 提取原 `FrozenTokenizer` 和原 `batch_plan`，不 import trainer 或运行旧 census。tokenizers 0.23.1 / Jinja2 3.1.6 沿原类严格检查；不 import Torch。

每行原 input IDs、labels、rendered prefix/full 四 SHA、完整 audit 字段、顺序与长度必须精确一致。总数固定 124 行、32 篇、87 患者、400614 tokens、395920 masked prompt tokens、4694 监督 tokens；最长 5556，6144 context 不截断。只生成位置与系数，不重写训练文件、answer、input IDs 或 labels。训练 r16、LoRA alpha32、2 epochs、batch1/accumulation8 及所有其余参数不在此修改。

列表值的字符范围复用原 census 的 compact JSON cursor。只按原生完整 BPE offset 判定相交，记录完整 label 索引 `q` 及 causal logit 索引 `q-1`；空 `[]` 也必须至少有一个值 token。token 跨 value/key 或边界标点时保留整个 token；跨两个患者的 value 时立即拒绝，禁止按字符比例拆 CE。原 token-loss 一半保留所有监督结构、key、EOS 与尾换行。

固定每 group 的目标为：

`L_g = .5 * mean_r L_old(r) + .5 * S/P * sum_r sum_p meanCE(value(p,r)) / n_p`

这里 P=87、每 epoch 的 S=16；`n_p` 是该患者本 epoch 出现在原提示中的次数，与 prepared 正例数无关。沿旧 trainer 的 `loss / len(group)` 反传，row 内写成：

`.5*L_old + .5*S*group_size/P * sum_p(meanCE(value)/n_p)`。

两轮原 shuffle seeds 20261023、20261024，每轮 15 组8行＋尾组4行，直接复用原 `batch_plan`。用 Fraction 逐 value token 累加，在每轮 S 个 group objective 的平均下，每患者新增分量严格为 1/P，乘 α 后 1/174；尾组不会再错误多除4。这个系数合同不宣称梯度、Adam/clipping 后更新或泛化改善，也不等于患者的完整 gold 正例负担。

外部输出固定为 `R/research/patient-balanced-mixed-loss-alpha05/preparation`：

- `row-weights.jsonl`：原顺序的124行；`native_audit`、`sequence_length`、`prefix_length`、所有 `supervised_label_positions` / `supervised_logit_positions`；`patients` 每项有原列表 `patient_index`、opaque key、`n_p`、value 两种位置、token数、prepared-empty标记及分数1/n_p。空 prepared value 不代表完整 gold 为空。
- `epoch-groups.json`：原两轮分组、`group_size`/`outer_group_divisor`，以及各 row/patient 在外层 group 除法之前的精确新增均值及 token 系数。分数字段统一为 `{numerator, denominator}`。
- `patient-component-contract.json`、`patient-counts.json`：逐 epoch/患者的精确系数恒等式及 n_p；没有按正例数加权。
- `manifest.json`：实际输入/源码/输出 SHA、native 完全一致检查、版本与耗时。只有实际成功执行后才生成；当前源码计划准备不声称124行派生已通过。

CPU 最长120秒；任何不一致立即保存失败并停止，已存在的 preparation 目录拒绝再次运行。无 heldout gold、旧预测、NN、fit 或质量评分。该固定α只是一个后继单变量控制，不扫描权重或更换保留门。
