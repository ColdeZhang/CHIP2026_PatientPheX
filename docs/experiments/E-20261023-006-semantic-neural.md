# E-20261023-006：神经语义概念识别与患者图传播

主分支汇总副本：来自 `day1/semantic-neural`，文档版本 `c9a7423b21101d5abfd5c2e877420e669dc50267`。实现及下列 CLI 位于该实验分支；平台接收状态以[统一提交凭据](../submissions/day1-20261023.json)为准，本报告不收集官方分数。

状态：首日实验完成。两个encoder独立验证完成，选择SapBERT；全量B预测已核验，主线程回报平台已接收，未读取官方分数，GPU已释放。分支 `day1/semantic-neural`，起点 `ac8a752`。目标是提供区别于精确词典、模糊排序、监督 BIO、病例检索迁移的首日方案。

## 假设与首版设计

冻结的语义编码器能将未见过的表型短语映射到指定版本 HPO；在训练折拟合小型 span 判别器过滤泛化词/背景描述，再将提及证据沿患者指称的句图传播，可以获得可提交的独立方案。新 HPO 本体和训练折实体构建候选词表，但最终候选识别与不精确链接使用 dense embedding，而不是字符匹配分数。

主版本模型为 `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`，约110M参数，revision `090663c3ae57bf35ffe4d0d468a2a88d03051a4d`，CLS表示与L2 normalization。MiniLM (~22M、revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) 的量化ONNX后端仅保留作CPU可行性对照，当前没有用其报告独立效果。HPO v2026-06-23，限制非 obsolete 的 HP:0000118 分支。训练数据只用共享训练折；不使用外部人工标注表型数据集、归档词典/代码/输出或验证标签构建资源。

预设首版：候选由新本体名称/同义词、训练折别名和医学词素结尾 n-gram 产生；语义表型判别器只拟合训练折；新词用最高余弦概念链接，精确匹配采用相同新版词表的概念映射。患者归属使用句间衰减传播，不把同段所有概念机械分给所有患者。首版阈值在运行前固定，验证后仅修正实现错误并记录。

## 已调研与取舍

| 来源 | 核验内容 | 采用或暂缓原因 |
|---|---|---|
| https://github.com/ncbi-nlp/PhenoTagger | 官方 v1.2 README：词典+本体远监督神经识别，无需人工表型标注；提供重建本体词表和训练入口 | 借鉴整体结构；现成模型为 HPO2023-10-09，不能直接假定适配2026-06-23；TensorFlow依赖较重，本轮先不下载其旧分类头 |
| https://huggingface.co/cambridgeltl/SapBERT-from-PubMedBERT-fulltext | UMLS2020AA 自对齐+PubMedBERT，CLS医学实体表示，Apache-2.0 | 主线程分配A30后选为正式首版encoder；不微调其基础参数，仅训练当前比赛span判别器 |
| https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | 通用句对对比学习、384维、Apache-2.0；官方提供量化 ONNX | CPU对照采用，其训练集不属于外部人工表型标注数据集；模型卡列出的通用句对资源与本任务限制区分记录 |

公开网页已于2026-10-23实际读取，研究原文与模型文件保存在仓库外研究/模型目录；小型来源及哈希见模型 manifest。未复制第三方源码，使用公开模型及标准依赖。

## 环境与最小验证

GPU已由主线程分配：10.5.2.3 的 A30 GPU0（24GB），开始前核验无计算进程；CUDA_VISIBLE_DEVICES固定对应UUID。服务器仅使用项目 semantic-neural 子目录，独立venv只读继承既有torch/transformers。默认CPU单线程。独立 venv 位于 `/workspace/patientphex-runs/day1-20261023/semantic-neural/venv`。

2026-10-23 CPU smoke：ONNX输出为 `(batch, tokens, 384)`；320 个短语（10批×32）单核耗时约0.213秒。该检查只验证资源可行性，不是效果评估。

共享64/16划分与无标签B已由主线程重新取得并冻结，使用 `/workspace/patientphex-data/day1/` 中对应文件；模型拟合只接触train64，验证输入无金标。共同评测器为新 `scripts/evaluate.py` 的公开规则近似，指标不称为官方分。

## 运行与结果

2026-10-23 运行过程：远端直连 Hugging Face 报 `Network is unreachable`，IPv4连接和 hf-mirror 也超时。因此改为本机下载固定版本，再由 MCP 分片传输；大请求曾触发 HTTP413，384KiB 原始块转base64成功。远端独立venv只读继承既有torch2.7.0/transformers4.50.0，新装sklearn1.6.1和pyahocorasick2.3.1，不改共享环境。该环境初始化与网络失败不会计为模型效果失败。

SapBERT原始权重437,955,508字节，使用 `prepare_model.py` 的numpy/safetensors转换为FP16副本218,990,904字节，派生SHA-256为 `dc5e5071ed867afd44cc2fdb21974f2c4ca2288442ac765be353eec80c2e4814`；转换不改变训练内容。下载期间GPU未占用，后续根据两路线模型到位时间协调交接。CPU MiniLM 验证先完成42,575条本体/训练别名编码，单核耗时71.153秒。

外部目录根为 `/workspace/patientphex-runs/day1-20261023/semantic-neural/`：`dev-minilm/` 为CPU对照，`sapbert-model/`为原始权重，`sapbert-fp16/`为转换权重；远端位于10.5.2.3项目挂载的 `semantic-neural/` 子目录。权重、缓存和完整日志均不进入Git。

### MiniLM CPU独立验证

| run | menF1 | docF1 | micF1 | macF1 | Score | 说明 |
|---|---:|---:|---:|---:|---:|---|
| dev-minilm / evaluator v1 | 0.690076 | 0.703665 | 0.380818 | 0.291784 | 0.516586 | 16文献36患者；公共规则近似分，非官方 |
| dev-minilm / evaluator v2 | 0.690076 | 0.703665 | 0.380420 | 0.297210 | 0.517843 | 主线程统一修正合法HPO复合关系项拆分；预测不变 |

完整运行127.927秒，单CPU线程，观察RSS约585MB（未做峰值采样）。训练判别器使用4,744个正跨度、19,132个负跨度，64篇训练文献，42,575个概念表面形式。预测859个实体、262个患者概念项，格式检查零错误/警告、覆盖全部患者。提及precision/recall为0.789290/0.613020，概念为0.850633/0.600000；v2关系precision/recall为0.519084/0.300221。公共评测器v2修复来源main `92fa1f8`，本分支cherry-pick为`3afb103`；保留原始v1和v2，后续比较统一使用v2。

预测SHA-256 `4b9e6b45866612d1dc3982d37a42bbaf47cd073292d35e9904d964dbee3fc0c2`，111,915字节。小型原始指标、验证、manifest、误差诊断与源码溯源更正保存在 [evidence/dev-minilm](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/c9a7423b21101d5abfd5c2e877420e669dc50267/solutions/semantic_neural/evidence/dev-minilm)。实际加载代码为 `ed34d40` 的run.py，SHA-256 `0a004a9c81bc1f12a582b39dd99b50ab3714a4d73d6396d47cf4e9dea34046ba`。过程中只扩展模型缓存签名，原程序结束时才计算源码hash，导致原manifest记录的是修改后磁盘文件；已保留原manifest并单独写溯源更正，后续代码改为入口即快照源码hash，长作业另留代码副本。该改动未改变预测逻辑或阈值。

误差诊断按纯HPO复合项展开后：36名患者中12名没有预测概念；317个关系漏报中159项在文献中已识别但未分配，158项来自实体概念缺失；126个误报中84项为金标文献概念分错患者，42项为额外概念。多患者家系表格文献12126309尤其差，多名患者的11个金标项中8个概念已识别但没有对应分配。**当前关系图不能可靠解决表格/共享家系与远距指代**；一次结果不否定语义编码器，后续主要应改关系归属。今天不据此再调验证集阈值。

复现命令（根目录按环境替换）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python solutions/semantic_neural/run.py \
  --train /workspace/patientphex-data/day1/train64.jsonl \
  --input /workspace/patientphex-data/day1/dev16.input.jsonl \
  --hpo /workspace/patientphex-data/hp.obo \
  --model-dir /workspace/patientphex-runs/day1-20261023/semantic-neural/model \
  --encoder minilm-onnx --device cpu --threads 1 \
  --output /workspace/patientphex-runs/day1-20261023/semantic-neural/dev-minilm/predictions.jsonl \
  --work-dir /workspace/patientphex-runs/day1-20261023/semantic-neural/dev-minilm
```

额外机制诊断：859实体中769项表面词在训练折/本体词表内，90项来自新表面词的dense检索；所有候选均经过语义span判别器。新词示例中 `thinning of the entire retina`、`color vision defects`、`abnormal movement`、`lack of coordination` 命中金标，但 `visual acuity`、`retinal arterioles`、`basal ganglia` 为误报。说明语义链接能补充表面变体，也会将正常解剖结构误映射为异常概念；不能仅凭余弦高就判定存在表型。诊断未用于修改本轮模型。

### MiniLM全量B备用

验后未改阈值，使用train80重新拟合：80文献、5,828正跨度、23,932负跨度、42,849语义词表项。命令将上述train/input/output/work-dir分别改为train80、B.input、B-minilm/predictions.jsonl和B-minilm；实际执行代码副本为B-minilm/run.snapshot.py，SHA `b1005687bc1162319aa5c87ab52fb6f8ef96ed8ed8518ea8295af4454c60647a`。

总耗时247.556秒；预测100文献244患者、5,394实体、1,820关系项。`validate_predictions.py`检查零错误/警告。文件703,874字节，SHA `810b45a984bc6e4ffec611c62f64c9a111e268bb1af1cec579107a5c1738f3c8`，位于外部根目录`B-minilm/predictions.jsonl`；manifest与验证报告在 [evidence/B-minilm](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/c9a7423b21101d5abfd5c2e877420e669dc50267/solutions/semantic_neural/evidence/B-minilm)。依赖真实版本在 [environment-cpu.json](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/c9a7423b21101d5abfd5c2e877420e669dc50267/solutions/semantic_neural/evidence/environment-cpu.json)。

### SapBERT GPU独立验证与选择

SapBERT权重中转完成后，远端组装并再次核对SHA；GPU0在开始前无计算进程、24,160MiB空闲。固定模型revision见上，原始FP32权重SHA `a4696930afef9aab296196d3d2142216c44cba24f21b4f285ceca7af21025614`，FP16派生SHA见模型准备过程。模型及参数未微调；阈值与MiniLM一致。实际运行代码为 `635fdd2` 的run.py，SHA `b1005687bc1162319aa5c87ab52fb6f8ef96ed8ed8518ea8295af4454c60647a`；后续文档提交没有改变该文件。

GPU dev作业 `proc_48396640-0da6-4116-a839-350eeb622c27`，2026-10-23 11:27:34 UTC开始，实际15.077秒，训练64篇、4,744正/19,132负跨度、42,575概念表面形式。完整命令与模型manifest保存在 [evidence/dev-sapbert](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/c9a7423b21101d5abfd5c2e877420e669dc50267/solutions/semantic_neural/evidence/dev-sapbert)。与MiniLM相比，除encoder之外运行后端/精度和依赖环境也不同：GPU torch2.7.0+cu128、transformers4.50.0、numpy1.26.4、sklearn1.6.1，因此这是实际方案对照，不声称严格隔离了所有数值库影响。

| 同一64/16划分、公共v2评测 | menF1 | docF1 | micF1 | macF1 | Score |
|---|---:|---:|---:|---:|---:|
| MiniLM CPU对照 | 0.690076 | 0.703665 | 0.380420 | 0.297210 | 0.517843 |
| **SapBERT GPU首版** | **0.701621** | **0.738916** | **0.429907** | **0.310659** | **0.545275** |
| 独立方案3 CRF参照 | 0.732540 | 0.745303 | 0.648472 | 0.612620 | 0.684734 |

SapBERT相对MiniLM提升0.027433，四项均升，选择它作为本路线唯一提交候选。相对CRF仍低0.139458，尤其micF1低0.218565、macF1低0.301962；不能将医学encoder优于通用encoder表述为神经路线已优于其他基线。CRF指标读取其 `crf-dev-v1/metrics.eval-v2.json`，不是复用其模型、代码或预测。

dev输出992实体、296关系项、16文献36患者，格式检查零错误/警告。文件129,277字节，SHA `c5eb86a5bcf894e7ab5d2812ca8d62d386733951e762faf83e8577a049a0de1f`；本地回收后再次核对hash。语义新表面词179项，其中80项匹配金标跨度/ID；其余新词包含正常解剖名词误报。12名患者仍无预测，292个关系漏报中164项概念已识别但未分配、128项实体概念缺失；135误报中93项是文献已有金标概念分错患者、42项为额外概念。只读诊断入口为 `analyze_errors.py`，没有用诊断结果再次调阈值。

dev结束后已复查GPU0无本路线进程，并把时段交给sequence。sequence完成后，11:29:58 UTC再次复查空闲，启动完整train80→B，作业 `proc_f38f6ea6-1df4-412c-aee9-ac134a9a46b6`。该路线只在主线程完成平台提交后才能称为已提交；本代理不操作网站。

### SapBERT最终B候选与结论

完整train80重新拟合与B推理共25.531秒，使用5,828正/23,932负跨度、42,849语义词表项。输出100篇文献、244名指定患者、6,143实体与2,024关系项，格式核验零错误/警告。实际代码、模型、阈值与SapBERT dev一致，仅训练数据增为全部80篇、推理输入改为B；对应完整命令见 [remote-B-job.json](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/c9a7423b21101d5abfd5c2e877420e669dc50267/solutions/semantic_neural/evidence/remote-B-job.json)。依赖真实版本见 [environment-gpu.json](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/c9a7423b21101d5abfd5c2e877420e669dc50267/solutions/semantic_neural/evidence/environment-gpu.json)。

最终文件 `/workspace/patientphex-runs/day1-20261023/semantic-neural/B-sapbert/predictions.jsonl`，801,072字节，SHA-256 **`f12d2ffb5a65edf16889d8f349a34f70c7fd3132cd145d6d385a123132d412d2`**。服务器持久副本位于10.5.2.3挂载项目 `semantic-neural/B-sapbert/`，本地回收与服务器校验一致；模型和embedding缓存仍留仓库外。小型证据在 [evidence/B-sapbert](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/c9a7423b21101d5abfd5c2e877420e669dc50267/solutions/semantic_neural/evidence/B-sapbert)。该文件已交主线程作为第五路线候选，官方接受状态与成绩应查主线程提交记录，不能从本地分数推断。

2026-10-23 11:30 UTC后检查：两个GPU作业均 `running=false`，GPU0没有计算进程；未修改或终止其他卡的进程。完成状态证据为 [remote-job-completion.json](https://github.com/ColdeZhang/CHIP2026_PatientPheX/blob/c9a7423b21101d5abfd5c2e877420e669dc50267/solutions/semantic_neural/evidence/remote-job-completion.json)。未测GPU显存峰值。GitHub认证在中间检查点曾失效，主线程已处理恢复；不能将短暂推送失败称为已备份，最终Git检查点由主线程台账统一记录。

**结论**：本轮医学语义编码器比通用MiniLM有效，提供了词表外表述识别能力；现阶段整体成绩受患者关系图拖累，明显弱于同划分CRF路线。作为独立首版提交以收集官方反馈，下一轮优先改共享家系/表格患者分配、跨句指代与正常解剖名词过滤；今日不继续试阈值，不追加其他外部人工表型训练数据。

### 提交交接

主线程独立复核dev四指标和B格式，得到的JSON报告与本路线保留报告完全一致，因此不重复入库。主线程于2026-10-23回报平台已接收 **`day1_20261023_semantic_sapbert_v1_f12d2ffb.jsonl`**，页面时间19:32:27，额度1→0；源冻结提交`77597ae`的run.py与实际执行快照SHA一致。未读取官方分数。本条来自主线程的提交回执，统一证据与五路线清单见 [首日计划](../campaigns/day1-20261023.md)，本代理没有操作网站。

## 后续官方成绩回收（主线程增补）

已按用户指令回收：Score **0.578946259**；menF1 0.645265、docF1 0.728051、micF1 0.519531、macF1 0.422937。score ID `3052947`、file ID `2795237`、seasonId `1862`；已与本报告冻结 B 文件 SHA、平台文件名/时间/大小核对一致。

这是原报告后的官方补充，未改模型或重跑。完整精度与提交标识见[官方证据](../submissions/day1-20261023-official.json)，横向结论与下一轮建议见[官方分析](../campaigns/day1-20261023-official-analysis.md)。
