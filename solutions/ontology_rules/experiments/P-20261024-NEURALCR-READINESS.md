# NeuralCR 完整库存与原生接口准备

目标是核实一个不同于现有三种 PhenoTagger head 的公开表型抽取资源能否合法、完整地加载并给出准确字符坐标。本阶段不读取赛事输入或标签，不产生质量预测，不使用 GPU。官方来源、训练谱系与 PhenoLinker 有界检索见 [只读研究](../public_neuralcr_readiness/readonly-inventory/review.md)。

固定 `data-team-uhn/NeuralCR` release `1.1`、源码 `1af8851b5e4a1ebeec7ced6c2f50d2dd34d4db0f`。ModelScope 准确身份接口没有可核同 revision 副本，采用官方 `HP.tar.gz`（583,483,718 字节）及 `pmc_model_new.bin`（1,345,465,351 字节）。上游未公布摘要，本地完整 SHA 将在实际下载完成后计算；小 Range 证据不是完整文件校验。

当前生产链已含一个 full64 LoRA，总参数 **9,597,467,990**，余 **402,532,010**。完整 fastText 输入词表及子词矩阵加 output 矩阵共 **333,363,400** 参数；CNN/10 ensemble 的实际 H5 形状待完整文件。默认用完整库存判断是否只能替换 BioBERT 的 121,587,826 参数，不因推断可能未使用 output 矩阵而先行扣除。任何后续裁剪必须是单独明确的部署契约，当前未实现。

## 实际准备状态

- [原下载计划](../public_neuralcr_readiness/download-plan.json)和源码于 `13e45d1` 推送后启动。2026-10-24 00:20:44.810536Z，A100 私有 `public-neuralcr-readiness`，`proc_1d81e43a-1d9a-458d-828a-ce1e6a725ef0`，600 秒截止 00:30:44.810536Z，TERM30；下载不使用 GPU。两资产分别串行 Range，合计两网络线程。原 `.part` 不覆盖。
- [CPU 依赖准备](../public_neuralcr_readiness/runtime-preparation.json)实际 6.786 秒通过，耗时包含小 wheel 网络、安装和 import。仅将精确 SHA 的 fasttext-wheel 0.9.2 装入自有 overlay；复用只读 TF2.12/h5py，GPU 列表为空，三个 TF1 compat 符号存在。没有加载模型或前向。源码与计划在 `a133cc9` 事前推送。
- 00:24:50 的[进度证据](../public_neuralcr_readiness/download-progress-00-25.json)：HP 已写 18,874,368 字节、fastText 134,217,728 字节。后续块也变慢，不能仅解释为首次连接等待；旧程序没有分离握手/首字节/流时间。原 600 秒继续执行，不自动延长。
- [续传准备](../public_neuralcr_readiness/resume-preparation/resume_assets.py)只准备四路总并发、8 MiB 缺口 Range。主线程已授权独立 attempt2 **1800 秒+TERM30**；须先确认原作业退出，再绑定不可变 prefix 的长度/SHA、原 ETag、assetID，写入独立目录，最终全文件拼接核长度/SHA。当前没有正式 `resume-plan.json` 或新网络作业。[冻结准备协议](../public_neuralcr_readiness/resume-preparation/preparation.json)不宣称首轮按时完成，也不授权自动第三轮。0.240 秒本地合成[传输契约](../public_neuralcr_readiness/resume-preparation/transport-check.json)通过中途断流续写、ETag 变更拒绝和原 prefix 不变；它不证明真实远端吞吐。新 wrapper 将保存真实 returncode，原 MCP 仅暴露 running，不能凭超时推测其 rc。

## 完整文件到齐后的最小检查

首轮实际于 00:31:04.996494Z 核实 MCP stopped、原 PID 和精确 downloader argv 均无；[终态](../public_neuralcr_readiness/download-terminal.json)为未完整，原返回码未知，不推测 rc124。稳定 prefix 为 HP 26,214,400 字节、fastText 150,994,944 字节，共 **177,209,344** 字节；fastText 的最后状态记有 3 次网络失败。原 state/partial 保留，不能称首轮按时下载完成。

独立 attempt2 的[正式计划](../public_neuralcr_readiness/resume-preparation/resume-plan.json)绑定这两个 prefix 的真实 SHA/原 ETag，只下载剩余 **1,751,739,725** 字节；4 路总并发、1800 秒+TERM30，完整输出写入 `public-neuralcr-readiness/resume-attempt2/downloads`。计划和源码推送后才启动，原文件只读；没有自动第三轮授权。

`a672245` 推送/远端 exact 后，[attempt2 作业](../public_neuralcr_readiness/resume-preparation/managed-job.json)于 **00:31:54.858401Z** 启动，`proc_65a836bc-bd90-41ad-bede-7ff673224f54`。wrapper 实际时刻 **00:31:54.919701Z**，1800 秒截止 **01:01:54.919701Z**、TERM30 到 **01:02:24.919701Z**；[首次运行证据](../public_neuralcr_readiness/resume-preparation/initial-progress.json)显示已开始下载独立缺口块，未完整。检查远端 `resume-job-state.json` 的实际 rc/时间及 `downloads/complete.json`，不得重复启动或覆盖其目录。

[已冻结 CPU 计划](../public_neuralcr_readiness/cpu-preparation-plan.json)仅安全提取 `HP/ncr_weights.h5`、`HP/config.json`、`HP/onto.json`，拒绝链接与目录逃逸，不反序列化 pickle、不提取外部标注。核 H5 唯一对象和完整 fastText 字典/两个矩阵至精确 EOF；异常头结构先保留证据，不能把未确认布局直接交 C++ 或私自修改。

原生加载另限总 180 秒，四个事前固定非空合成文本，保持 10 ensemble、原 `.8`、1–7 gram、原 overlap 政策。预先允许的一行兼容改动仅 `tensorflow` → `tensorflow.compat.v1`；数学不变。检查实际权重恢复、概率有限、返回字符坐标与真实打分词串/原文的对应。不把接口通过当质量增益。首次故障保留，必要时仅一次有证据的接口修复；未完整下载/库存不足时不进入赛事质量。

## 独立 attempt3：本地 Range / MCP 小片中转

Attempt2 实际在 **00:36:10.551370Z** 以 rc1 结束，255.631 秒；最后异常在响应 payload `read`，不能称 1800 秒耗尽或断定只是 GitHub 重定向慢。无旧 writer、全部稳定片段逐 SHA 冻结，累计可复用 **248,512,512 字节**。完整块耗时 2.16–164.90 秒，未观察到稳定相同连接限速。[终态清单](../public_neuralcr_readiness/resume-preparation/terminal-inventory.json)保存原 prefix、7 个完整块及各 partial；不覆盖首两轮。

主线程另授权 attempt3 **3600 秒+TERM30**：[新计划](../public_neuralcr_readiness/local-relay-attempt3/plan.json)只补实际缺口，本机官方固定 ETag Range 8 MiB → MCP ≤512 KiB 写回 A100，单一总线程池 8；每片至多 3 次，保留所有成功片和本地 partial，单片失败不丢弃整批。HP 缺 485,966,150 字节优先补完，实际完整 SHA、安全提取与 H5 唯一 shape/count 后，只有替换 BioBERT 的完整参数投影可行才继续 fastText 1,194,470,407 字节缺口。保守仍计 input/output 两矩阵。两大完整文件到齐后才进行原定 180 秒合成原生检查。

[网络普查](../public_neuralcr_readiness/network-census/summary.json)中本机同官方 1 MiB GET 约 .41 秒，4×512 KiB MCP 写 3.846 秒、含服务器 SHA 共 5.207 秒；吞吐投影不是保证，不增加探针或公网 tunnel。[无网络传输 fixture](../public_neuralcr_readiness/local-relay-attempt3/transport-check.json)通过中断只补余字节、单片重试、512 KiB 限制与 HP 参数门阻止 fastText；源/计划先提交推送后才启动 bulk。

`00fa820` 已推送/远端核验后，attempt3 于 **00:44:22.681751Z** 启动，PID80371，截止 **01:44:22.681751Z**、TERM30 至01:44:52；[实际启动记录](../public_neuralcr_readiness/local-relay-attempt3/initial-job.json)。完整文件后的[CPU源准备](../public_neuralcr_readiness/postdownload-readiness/preparation.json)复用原完整fastText矩阵解析与原生检查，当前未启动模型加载。

### HP 完整资产与实际参数门

[实际主文件终点](../public_neuralcr_readiness/local-relay-attempt3/HP.tar.gz.assembly.json)已完成：583,483,718 字节，SHA `08e48710399ce8aa321367d6dafdef87a464048b85437552edc728e0e129668b`；938 个既有/新片段全部按位置和SHA核验再拼接。安全提取只包含 H5、onto.json、config，archive 没有单独许可文件；保留仓库 MIT 与权重未单列许可的区别。内部拼接/提取/shape 9.088 秒，[实际外层job](../public_neuralcr_readiness/local-relay-attempt3/HP.tar.gz.segments.json.job.json)另列。

实际 H5 630,900,616 字节/SHA `9a43dd5a237f7ad630ca26bed729aee07a8cc6c5cac9c90916be4ded9520d807`，**60 个唯一数组，157,705,490 参数**；10 ensemble，每个 width1 Conv1D 100→1024、Dense 1024、层级 head 14261，其中14260 ontology概念（含root），额外一列为原代码的 `None`。加两fastText矩阵的保守header投影为 **491,068,890**；直接追加 **10,088,536,880** 超10B，替换BioBERT为 **9,966,949,054**（余33,050,946）。fastText全文件shape仍未完成，不能把投影说成完整库存；原冻结门允许继续补词向量，当前不进行NN/质量。

### fastText 条件性接口兼容准备

固定 NCR README 要求 fastText≥0.9.1，所关联0.9.1序列化源码 `Dictionary::save/load` 都先写/读 int64 `pruneidx_size` 再逐词记录。先前官方prefix的首词 `the\0` 在84、`prune=-1` 在88，顺序与标准不合；现只准备[单次窄修源](../public_neuralcr_readiness/postdownload-readiness/header-repair-preparation.json)，未执行。必须先完整原件 SHA/大小、666817词条、两个矩阵和精确EOF证明此布局，才在独立派生副本交换这12字节，保持原件、词序/计数/类型、所有数值矩阵及[0:84]/[96:EOF]逐字节相同。不将未知格式交C++，不截断、重建词表或裁参。

当前授权仍仅下载/库存/180秒四合成native；将来若考虑替换BioBERT，必须从真正旧双head候选池重建、移除其来源及6字段并fresh HGB，不能保留依赖BioBERT的103列旧模型或缓存却扣除其参数。尚未开展这个质量控制。

01:03环境恢复后已核原PID80371/80372继续执行，同一截止与0失败收据不变，未重复中转；顶部台账已纠正过期attempt2描述。[CPU执行包装](../public_neuralcr_readiness/postdownload-readiness/execution-preparation.json)只准备逐阶段入口：完整库存120秒、条件性窄修120秒、原生四合成180秒，另各15秒TERM清理；实际stage plan须绑定完整资产SHA后先推送。shape/native复用同私有目录硬链接，不再下载或复制大模型；当前未启动任何阶段。

尾段传输遇到共享MCP故障，新只读请求实际返回 `LUNA_OFFLINE`；不是公网资产缺失。已逐SHA核全部203个已计划公网缺口range共1,680,436,557字节在本地完整，另外248,512,512字节旧prefix/片仍在原服务器。原relay按既定逐片3次/3600秒收尾，不自动重开；失败片与原参数来源均保留。[离线状态](../public_neuralcr_readiness/local-relay-attempt3/provider-outage-0111.json)和[本地完整range清单](../public_neuralcr_readiness/local-relay-attempt3/all-local-network-ranges.json)是恢复入口，不能冒称远端fastText完整或native已运行。

01:12实际新只读roundtrip已恢复，原attempt3自然结束为 `incomplete_fragments_preserved`（1681.128秒/12片共6,291,456B失败；非预算耗尽），原writer80372已无、timeout壳无命令字节。主线程授权仍在原01:44:22.681Z截止内的[独立MCP恢复](../public_neuralcr_readiness/relay-recovery-0113/recovery-plan.json)：先全2280个FT片/旧prefix实际SHA库存，仅缺失片用本地原字节写新目录，至多8worker/每片3次、≤512KiB；不重下公网，不重复HP。原失败state/receipt原留，后继完整资产manifest分别引用原HP与恢复FT，不篡改原attempt3终态。源/输入清单本检查点仅准备，尚未传输补缺。

恢复前完整远端库存实际3.546秒/rc0：2280段中2273段、1,341,795,335字节实核正确，只有7片共3,670,016字节缺失；部分原失败RPC实际已落盘，故不能直接按12个失败请求全量重写。报告首取回`UNKNOWN_OUTCOME`，仅再次读取同一已有报告成功，未重跑库存。实际[补缺计划](../public_neuralcr_readiness/relay-recovery-0113/recovery-bound-plan.json)只写这7片新目录；对未来managed start先落intent，若响应不确定先核原进程，禁止盲目重启（H32）。

已冻结7缺片补缺启动后，连接再次离线：110.265秒/21次实际LUNA_OFFLINE/0补写成功，无assembly，原state与本地range保留；不把失败请求视为未落盘，恢复时仍须重新库存。主线程统一确认连接前不再启动writer/native。并行[纯输入候选普查](../public_neuralcr_readiness/input-only-census/plan.json)从原生源AST逐字句提取NN前枚举，6合成坐标合同通过；首跑仅因24个已有passage含title元数据被过严allowlist拒绝，失败源/plan保留，只放行该元数据后重跑CPU，不改枚举/输入/门槛。
