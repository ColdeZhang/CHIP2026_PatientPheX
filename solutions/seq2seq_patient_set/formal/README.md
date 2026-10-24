# 唯一正式 LongT5 配方

固定原base fresh全参训练10epoch×87患者，batch1/accum4、每epoch尾3用真实组均值，220次Adam。lr1e-4、wd.01、betas(.9,.999)、eps1e-8、clip1、HF线性warmup22、seed20261024；标准scheduler首步LR0会推进Adam状态，逐步记录lr_used/state.step，不把首步称权重已变化。全输入/目标不截断，不pack、不挑epoch。

[plan.json](plan.json)固定全部核心源、原87/39供给、支持控制和双fold条件。[formal-cost](../evidence/formal-cost.json)使用真实四形状+全87 tokens：训练及最终完整stateACK独立1800秒+TERM30，39推断另行锁时钟；保守代理1448–1667秒，不称实证上界。

入口 `driver.py --plan plan.json --binding <新排他binding.json>`。binding必须有root实际资源协调、当时300秒有效期、准确GPU UUID/模型目录/供给目录/完整CPUreceipt/输出与driver目录、Torch/TF版本和预算；driver在创建子进程前核TTL，runtime在载模型前再核，fresh GPU≥20000MiB且无同卡进程。UNKNOWN先恢复原意图，不重复start。当前无formal binding/启动。

同host `.100` 已齐全部资产/87训练/39纯输入，根 `/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/seq2seq-patient-set-formal-v1`；base复用相邻 `seq2seq-patient-set-v1/assets-base`，不复制990MB。Python ScratchVL3.10.12；PYTHONPATH依次本root `runtime-sp`、旧root `runtime-t450`、项目 `case-retrieval/round2/training-runtime` 与 `case-retrieval/round2/runtime`。原生SP.2.1/TF4.50/Torch2.7.1cu126，CPU实际[完整合同](../evidence/formal-native-CPU.json)无CUDA。`.3`复制因离线失败，不能称新host已就绪。

`state_io.py`复用BioClinical已验fullstate写入/比较/恢复序列，保存model、Adam、scheduler、Python/NumPy/Torch CPU/CUDA RNG、访问次序、计划/源码。完整文件fsync+父目录fsync后reopen逐张量比较，再restore全状态并复核共享输入同Parameter、head独立、optimizer参数/状态唯一。实际tiny CPU120张量/29状态通过；完整base约3.17GB及其真实ACK必须在1800秒内产生。只最终epoch10，probe状态丢弃，不称恢复轨迹等价。

`infer.py`只能加载完整220Adam/10epoch最终state与ACK，严格新model load且输入共享/head独立。完整39 greedy/cap1600，先保存原all IDs/去唯一decoder-start后的suffix与结束状态，再原生SP+固定32099小于号传输解码；原生UNK不修文字，未定义extra/padded ID raw保留并拒transport。`extract.py`独立验证T5实际raw链，再调用原G `interpret`；不得伪称Qwen HF_decode_equal。

合法quotes继续原G `held_prepare.py`/必要public Sap/`held_finalize.py`与b59 policy，保持train32 alias/public41099，不拟合、不扫cut。`verify_normalization.py`仅机械复用旧G verifier后半段，T5 raw前半段由extract另核。唯一终点为B4∪[T5∩(G_old∪S∪Q∪C)]：F1 B4 .701325377212172，133条B4外固定支持；完整预测freeze后最多1new-v2，旧0/无改变0。F1严格正才同方法F0，双折正才提出full64；macro完整披露。F0未准备模型/数据/epoch计数，不拿F1 87患者直接复用到F0。
