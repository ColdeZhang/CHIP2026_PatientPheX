# E-20261023-004：监督序列标注首版

状态：进行中；分支 `day1/sequence-crf`，基础版本 `ac8a752`。主线程统一筛选与提交，本路线不操作比赛网站。

## 目标与假设

以训练折监督的 BIO 序列标注学习表型边界，再链接到指定 HPO 并判定患者归属。假设：词形、局部上下文和本体词表特征能够学习字面词典之外的变体，同时减少词表中的非表型误报。与其他路线的主要区别是使用线性链条件随机场学习完整标注序列。

只使用主线程提供的 64 篇训练折；验证 16 篇文献仅用于独立评估。若首版合格，再用完整训练集重拟合 B 预测。禁止使用旧归档词典/实现，禁止把验证金标用于建词典。

## 已查阅的成熟方法（2026-10-23）

- [sklearn-crfsuite 官方教程](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html)：词形、前后词上下文与 L-BFGS 弹性网训练。复用软件包，独立实现本任务特征与输出。
- [BiomedBERT 模型卡](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)：MIT；从 PubMed/PMC 无标注文本预训练，作为有 GPU 后的监督 token-classifier 升级候选。已核验模型卡；权重获取状态见 GPU 升级记录。
- [GLiNER 官方仓库](https://github.com/urchade/GLiNER)：成熟零样本/可微调 NER 框架；具体 checkpoint 的训练监督资源尚未核验，因此本次不直接使用未知来源权重。
- BioBERT v1.2 的 Hugging Face README 请求返回 HTTP 错误，未据此断言模型不可用。

先完成可复现 CPU 首版，再在主线程分配的独立远端目录/GPU 时段比较预训练模型。

## 计划实现

1. 按原 passage offset 保留全局坐标，BIO 标签来自训练金标；明确处理重叠实体和否定项。
2. CRFsuite 词形/邻词/本体词表特征，固定首版超参数。
3. 仅训练折别名 + HPO v2026-06-23 概念链接；所有输出 ID 限于 HP:0000118 分支。
4. 使用患者给定指称与局部上下文做归属，完整覆盖指定患者，包括空集合。
5. 用主线程统一新评测器报告四项近似分数，并执行坐标/ID/患者覆盖校验。

## 环境与产物

自有环境 `/workspace/patientphex-runs/day1-20261023/sequence-crf/.venv`；大产物位于同一路线目录的独立 run_id。CPU 线程限制为 1，不写入权重、全量预测或日志到 Git。

首个最小自造病例 smoke 已通过：原始非零 passage offset、末尾实体边界、无效 HPO 排除、指定患者覆盖，以及输入中恶意注入 entities/association 不影响预测。CRF 训练 3 轮，1 文档 10 token，端到端约 2.71 秒；这只是工程检查，不是验证分数。指定本体解析得到 19,120 个概念、42,567 个名称/同义词（含该自造训练别名）。

正式数据划分与评测现已就绪，已 cherry-pick 共享基础提交 `90d52d0` 到本分支（本地提交 `cdcdb3e`）。

## 首个独立验证结果：CRF v1

训练 64 篇、验证 16 篇/36 位患者；由主线程统一提供划分。仅训练折建立 44,005 条链接名称；序列训练含 211,596 token、10,871 个片段；对齐 4,713 个金标，32 个重叠金标按最长优先舍弃（线性 BIO 无法同时表示重叠实体）。

| menF1 | docF1 | micF1 | macF1 | Score |
|---:|---:|---:|---:|---:|
| 0.732540 | 0.745303 | 0.650549 | 0.614222 | 0.685654 |

全部为 `patientphex-public-approx-v1` 的本地独立验证近似分数，不是官方成绩。实体 P/R=0.8561/0.6401，关联 P/R=0.6393/0.6622。主要瓶颈是实体召回与患者归属误报；不能靠同义词全量匹配推断序列模型已充分覆盖。格式校验零错误/零警告，16 篇输出 827 个实体、463 条关联。

命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /workspace/patientphex-runs/day1-20261023/sequence-crf/.venv/bin/python solutions/sequence_crf/run.py --train /workspace/patientphex-data/day1/train64.jsonl --input /workspace/patientphex-data/day1/dev16.input.jsonl --hpo /workspace/patientphex-data/hp.obo --output /workspace/patientphex-runs/day1-20261023/sequence-crf/crf-dev-v1/pred.jsonl --work-dir /workspace/patientphex-runs/day1-20261023/sequence-crf/crf-dev-v1 --iterations 80
```

拟合 21.80 秒，推理 1.06 秒。预测 SHA-256：`b7bcdeabc23f8d5915f2d1b5f9b6631b7565c66d78a60460fc405aee3286942b`。关键指标、格式校验和输入/产物校验值已保存到 `solutions/sequence_crf/results/`；全量预测/模型留在外部产物目录。

工程修复：首次用 CLI 保存自定义 Linker 对象后，从另一个 Python module 读取触发 `__main__` 类引用问题；已改成保存纯属性状态，跨模块恢复、坐标和输入标注隔离检查实际通过。该问题未改变预测算法。

## GPU 升级准备

主线程提供 A30 24GB 轮转时段。独立远端目录 `sequence-crf/`，新建自己的 venv，以 `--system-site-packages` 只读复用现有 torch 2.7.0+cu128 / transformers 4.50.0；只向新 venv 安装依赖，未修改原 conda。已实现 BiomedBERT 监督 BIO 首版：384 token、stride 64、4 epochs、batch 16、lr 3e-5、fp16；链接/关系头与 CRF 相同，用于隔离编码器贡献。当前等待权重下载及主线程释放指定 GPU，尚无该模型验证结果。

## 全量训练与 B 预测：CRF v1

通过验证门槛后保持超参数不变，改用 `train80.jsonl` 拟合，并对 `B.input.jsonl` 推理。100 篇/244 位患者输出 5,229 个实体、2,868 条患者表型关联；拟合 21.12 秒、推理 8.90 秒。格式校验 0 错误、0 警告。

产物：`/workspace/patientphex-runs/day1-20261023/sequence-crf/crf-B-v1/pred.jsonl`；SHA-256：`8014f006eb8b90103172838ed7d8b1a3c85b681ef68b63c98200096e8112ae8f`。此文件为可靠候选，主线程将在 BiomedBERT 对比后选本路线一个版本提交，避免消耗两次名额。

两次 CPU 正式运行使用 Python 3.11、sklearn-crfsuite 0.5.0、python-crfsuite 0.9.12、scikit-learn 1.8.0、numpy 2.3.5、scipy 1.17.0。保存于 manifest 的实际算法源 SHA-256 为 `0222778730036395a424fa7f8d83cd80649d25cd723a2ffc803109e433327d05`；后续提交加入运行库版本记录，不改变模型算法。

远端获取 BiomedBERT 初次尝试失败：`huggingface.co` 报 `Network is unreachable`，尚未启动 GPU 训练。正在检查镜像/可验证缓存；该失败属于资源获取问题，不据此淘汰技术路线。

BiomedBERT 权重来源已固定到 revision `e1354b7a3a09615f6aba48dfad4b7a613eef7062`。原始 `pytorch_model.bin` 为 440,474,434 字节，官方 LFS SHA-256 `ad7bbb66376cfd6b2db3447192b034efe016337cbef135c35c411fd61b13c193`，本机下载校验一致。远端模型站点均不可连接，改本机中转；MCP `write_file` 请求大于约 1 MB 会返回 413，改为 512,000 字节原始数据分块，经 base64 上传后在自有目录合并校验。所有数据在远端解压后与共享源 SHA-256 一致，未传验证金标。远端裸 `python` 命令不存在，应始终使用项目 `.venv/bin/python`。
