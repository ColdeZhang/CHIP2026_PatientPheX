# MedGemma 1.5 4B：严格同池关系控制准备

当前：原始19件资产与实际883个权重header已核；.100原生CPU合同一次通过，尚无GPU或质量结果。A100源码包部署遇MCP工具错误，root统一确认A100离线后改用A800独立SHM；原未启动传输计划及未知write库存保留，新A800/发送端小源码及数据包已actual SHA核验，HTTP模型复制尚未开始。真实恢复先看[状态](preparation-status.json)，不要重跑已有GPU任务。

目标是以公开MedGemma作为独立关系模型，与旧SFT32的F1开发Score .660942003比较；沿用原124条train32监督和80条纯输入，CID→PID schema、候选、患者与病例原文保持。只替换公开模型与必要原生模板，不先做并集，也不以旧9B响应补齐输出。不同架构、规模、tokenizer和初始化共同变化，不能把未来差值单独归因于医学知识。当前不读新gold，不扩原dev/官方。

## 已完成CPU与来源

ModelScope固定`google/medgemma-1.5-4b-it@9bca96ca3e3049f6a084d8c876078cedb3069f18`，HF对应`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`。资产由score_gap原单次作业完成949.172秒，19件8,639,802,098B全SHA、两shard883BF16 tensor/offset/EOF核验；没有重复下载。原HAI-DEF条款/政策/NOTICE随资产保留，未来adapter含修改说明。小凭据见asset-references；权重/tokenizer大文件不进Git。

缓存Transformers5.20.2/PEFT.21.2中的原生Gemma3完整bundle，实际modeling SHA8ca9a7ad；模型卡记录4.57.1，未为版本号另装环境。保留全部4,300,079,472参数，虽然纯文本不调用视觉/projector，资源库存仍将其计入。只对34层×7语言投影（q/k/v/o/gate/up/down）加r16/alpha32/dropout.05 LoRA，共29,802,496；总单模型4,329,881,968。embedding与head共享且冻结，视觉/projector冻结并设置forward拒绝guard；不照抄视觉all-linear或modules_to_save。

**官方10B限制适用于每个独立模型**（overview L164–165、论文§3.1），不是全流水线参数相加≤10B。此次单模型核查通过；以后组合仍需单独预定质量控制及资源披露，本轮不据新规则增加模型或质量臂。

实际CPU10.453秒、CUDA未初始化、全部模型参数在meta：训练124行/32文献，共382000tokens、最长5380、监督7150；F1原80行/16篇共238727tokens、最长5422，587CID/1557患者候选对。全部无截断，原数据SHA与旧两16互斥OOF来源不变。源3daa85b0/interfaceaecef7ea，CPU7e6fb9dd，详见[完整计划](plan.json)及[统一源清单](source-manifest.json)。

原生模板把system并入首user（两个换行），assistant显示为model；完整target JSON保持，监督原生`<end_of_turn>`106及末尾换行，生成EOS固定[1,106]。原Qwen分词数402594/7170与新382000/7150不同是tokenizer结果，不删除样本或答案。native SDPA保留Gemma自己的混合sliding/full causal attention，不替换loss或kernel；原生TF5默认dynamic cache替代卡中旧v4的legacy hybrid配置，必须由实际2token接口核查。

## 唯一诊断与后续训练计划

root已条件授权：资产复制到A800独立SHM并核本机原生runtime差异后，等待sequence实际最终释放，再fresh核GPU2 UUID/占用，单次180+TERM30最长训练行更新；无具体源/CPU阻断时报告后可执行，当前尚未启动。诊断使用5380最长完整训练行，native.loss与同一次forward的有效位置手算CE核对（rtol/atol1e−5），有限梯度、实际非零LR更新attention/MLP B参数，冻结基座无grad。随后最多2个greedy token只查原生缓存接口；若首token为EOS，仅记录prefill，不冒称已走第二次缓存、不追加probe。全部诊断状态丢弃，不能作为训练或质量结果。

正式方案只作准备：fresh公开base，两轮原124/248曝光/32calls、accum8/尾4、seed20261023、lr1e−4、warmup2、AdamW与clip1、BF16/SDPA/非重入checkpointing，未授权启动。六个数据/调度/collate函数及原生loss微批循环AST与旧trainer一致。正式时间预算由上述唯一实测成本/峰值决定，不能先把更小基座当作显存或耗时保证。每完整epoch保存adapter＋Adam/scheduler/RNG，SHA/fsync及原子发布；初始新run不读取旧Qwen adapter或optimizer。

后续仅完整epoch2才按固定80输入greedy1600生成；新raw/fullpred先freeze，唯一公共v2 all/text对缓存旧基线，不同时试并集/阈值/F0/原dev。主目标为总Score提高，micro/macro及错误类型如实报告；这是反复观察的F1开发折，不是独立验证或官方成绩。

## 存储与连接恢复

.100完整模型：`research/medical-small-model-readiness/models/medgemma-1.5-4b-it-9bca96ca3e30`。原A100计划保留为superseded-unstarted；新A800权重/大状态只用自有`/dev/shm/patientphex-medgemma15-1045-20261024/`，项目只保存小源/日志。02:39:47只读实查A100项目剩58.139GB，现成Python3.10.21/Torch2.6.0+cu124/TF5.20.2/PEFT.21.2与Gemma源可用、CUDA未初始化。两个新完整epoch state按adapter＋两份AdamFP32约715MB，另有元数据；无需删除历史。

.100→A800固定19资产一次私网HTTP复制授权600+TERM30，逐full SHA/Range断点/fsync；复用旧已通过transport handler与逐文件fetch loop的AST，仅改固定资产manifest/namespace。原A100源数据489KB包经MCP准备失败，未启动HTTPserver/fetch；保留可能落盘的自有upload chunk，当前不反复查离线机器。A80002:49:16实际只读剩247.529GB SHM、171MB项目盘，Py3.11.17，Torch2.7.1/TF5.20.2/PEFT.21.2/tokenizers.23.1及Gemma源与.100原CPU一致。新SHM副本不是持久备份，.10019件完整原件保留。sequence已02:49:43完成最后fullstate导出解除带宽避让，GPU2仍保留给其最后80推断，收到实际最终release才使用。详情[新复制计划](transfer-a800/formal/run-plan.json)、[传输复用检查](transfer-a800/source-reuse-check.json)；8.64GB只走私网HTTP。


## A800 复制中断与唯一恢复（2026-10-24）

首600秒窗口因本地监控 MCP initialize HTTP503 中断；旧controller finally停止源server，fetch随后connection refused。两个原job均实际停止，10个小件完整、首权重2,091,970,488B断点保留。没有模型加载或质量结果。按实测剩余6.548GB和9.0–9.3MB/s，root在恢复启动前明确把原600恢复提案改为唯一900秒＋TERM30；原未启动600计划与首失败原样保留。

恢复仅从冻结part续传，10个已完成文件依原full-SHA后atomic rename和不变stat继承，不宣称本窗口重读；其余逐文件full SHA。19原件仍持久在.100，A800 SHM仅临时副本。监控失联现在只保留known job，不再finally打断可能健康的传输；远端绝对超时继续有效。详见transfer-a800-recovery。尚未启动GPU诊断或正式训练。


恢复已于03:14:31.612Z完整结束，实际722.052秒<900。19件8,639,802,098B：10小件继承原核验，9件本轮完整SHA；ACK逐项分列，源server和fetch均已停止。原首窗口失败仍保留。实际ACK绑定的单次A800 probe计划已冻结；45秒隐藏CUDA差异核后fresh空卡，180+TERM30最长更新及最多2token缓存诊断，尚不正式训练/评分。
