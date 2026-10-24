# Qwen3.5 原SFT32全文124×2：正式训练入口

只准备已授权训练与完整epoch恢复；本目录无推理/评分入口，也未执行远端或模型。fresh原Qwen3.5-9B/r16/seed20261023，124条F2+F3/32文献、原7170监督，原候选与目标全部保留；改动仅完整允许文章追加与此前通过数值/工程合同的selected CE。max13549、cap13824工程拒绝上限，不截断。

- 模型8,953,803,264+LoRA43,278,336=8,997,081,600参数；原12目标、alpha32/dropout.05、BF16/SDPA/原生GDN、gradient checkpointing use_reentrant=False。完整BF16 head保持，在FP32 CE前筛选有效监督位置。
- 原每epoch16组（15×8+1×4）、32次AdamW、warmup2、LR1e−4、wd.01、clip1、seed+epoch shuffle均保持。两轮原微批循环、batch/collate AST精确相同；不沿用420probe非零诊断状态。
- 第1进程必须`--stop-after-epoch 1`；完整state原子发布到固定私有SHM，退出释放卡。Controller持久复制全checkpoint和少量_run文件至.100，经SHA/fsync/最终文件重开校验，传回`export-ack-epoch-001.json`，才可第2进程`--resume <output>/epoch-001 --resume-export-ack <output>/export-ack-epoch-001.json`。
- 第2进程先核原fingerprint、全checkpoint/ACK/source/member SHA，再加载同public base和该完整adapter；Adam全部state、scheduler、Python/NumPy/Torch/CUDA RNG、global_step16和parameter顺序必须完整恢复。保存和恢复的全逻辑state digest、adapter tensor digest相同后才训练。CPU没有Torch，不能把stdlib反例当实际Tensor/GPU续训等价；运行中会另产resume-state-verified.json。
- 每进程传同一个`--absolute-deadline-unix`，整体8400+TERM30由controller硬限制，导出每轮≤600且占总预算。deadline在fingerprint内，恢复不得重置。trainer在开始、每group及checkpoint前后再检查。checkpoint始终在私有SHM，不落127MB项目盘；本地SHM fsync状态明确不等于持久ACK。第二轮training_complete仍标注等待末轮持久ACK。

CLI与准确输入/依赖SHA见training-plan.json。统一output：`/dev/shm/patientphex-qwen35-fulltext32-1045-20261024/formal2ep`。输入expanded124 SHA9e80e541；实际完整encoded0755d5fc、1009181tokens/13549/7170监督与已完成native逐行IDs/labels SHA每次在真实编码时核验。这里不重做native census或selected CE数值试验。

CPU-contract-result.json：69项stdlib新入口/来源/完整状态字段/缺失Adam与重置warmup/ACK/目录过滤/超期反例通过，0.028825秒；另有完整原微批loop AST精确收据。无GPU/远端/评分/Git动作，待owner checkpoint和统一controller执行。

H44防复发：仅`epoch-[0-9]{3}`真实目录可作checkpoint，export ACK使用独立前缀。进程恢复可更新run_manifest/progress等运行元数据（manifest身份本身须相同）；已经发出SHA的epoch checkpoint和.100 ACK不可修改，epoch1_run历史由持久包保留。
