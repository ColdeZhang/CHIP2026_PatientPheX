# DeepSeek-R1-0528-Qwen3-8B：原生资产、接口与成本

准备实验 `P-20261024-R1-0528-READINESS`。**准确模型资产及原生加载通过；1,024 token预算未产生完整最终答案。** 唯一GPU程序45.260秒、wrapper46.141秒/rc0，首条达到token上限后按事前规则停止余两条；无gold、全80推断或质量分，不据此否定R1提升比赛分数的可能性。完整[合一清单](terminal-manifest.json) SHA `d172a788…`，实际[小摘要](gpu-probe/terminal-summary.json) SHA `a1d3aea0…`；统一独立终审等待中。

## 为什么研究这个准确模型

官方overview §4.2 Agriv明确将Qwen3-8B与DeepSeek-R1-0528-Qwen3-8B生成的患者—表型证据做OOF/LightGBM融合。§3.1是每个独立模型≤10B，允许这种集成。它证明有成熟使用依据，不证明本项目边际收益。已有本地记录未发现这个准确检查点的任务实测/cache；普通Qwen3或9B thinking负结果不能替代它。[搜索范围](existing-resource-search.json)与[事前可行性计划](feasibility-plan.json)保留。

ModelScope固定 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` revision `5ee5fa4832e658ab84298806de06ed2a167834bd`，MIT代码/权重许可。最初只读HTTP Range头推导399 BF16 tensors、8,190,735,360参数；36层、4096宽、32Q/8KV、head_dim128、MLP12288、词表151936，embedding/lm_head不共享。完整模型BF16数值16,381,470,720B；参数量满足单模型规则。

## 资产与CPU的真实终点

.100自有持久目录单次2100秒+TERM30窗口下载：03:05:56开始，03:36:58完成，报告1862.200秒；9文件16,388,603,517B全SHA与固定来源一致，399实际头/连续payload/精确EOF通过。原managed `proc_1c596785-307e-4471-8541-c982943da8bb` 已观察running=false，无重试、无延期。下载报告elapsed止于小header检查前；没有伪造工具未提供的wrapper退出码。[完整资产收据](assets/complete.json) `5c7ed899…`、[实际头](assets/actual-header-inventory.json) `ef4303dc…`。

保留准确DeepSeek tokenizer/template，不能套普通Qwen的enable_thinking开关。实际80纯输入消息/候选/患者原样；原生token总269064，min852/median3425/max6002，True/False enable_thinking均被模板忽略且字节相同。CPU首版因TF5默认BatchEncoding与list比较失败4.791秒；仅显式return_dict=False后5.582秒完整通过。strict final解析器2正12负通过，要求自然EOS、完整可选think、全部CID键与合法PID；截断不是负关系。

YaRN原配置保留，包括非标准attn_factor；实际TF5与HF4.51原生函数均读取attention_factor而不读取该非标准字段，缩放1.138629436111989，64个CPU FP32频率逐位/bytesSHA相同。保留warning，不静默改config或推断其他引擎。首次CPU辅助源占位符语法错误原留，修复后真实通过。[CPU完整清单](cpu-terminal-manifest.json) `038d6044…`。

启动前主线程发现跨CPU/GPU字节强等可能误拒正常舍入。实际TF5源证明非持久buffer原生FP32，经meta搬运及原生函数重建；保存旧源，明确拒绝未知dtype而不cast，加载内比较同device原生64值，事前rtol=8×FP32 epsilon、atol=0。实际buffer与CPU bitexact；GPU函数参考最大绝对差1.86265e-9/相对差9.07048e-8，在约9.53674e-7容差内。没有独立额外GPU探针、模型/config/生成变更。[窄修依据](gpu-probe/rotary-prestart-change.json)。

## 唯一A30格式/成本探针

执行plan `dc3ba5f1…`、runner `796df0d7…`，.100 GPU1 UUID `GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539`。fresh卡0MiB、free24164、目标无计算进程；其他UUID作业未触碰。managed `proc_2eb8109a-602c-4714-956c-1aa4db401a35`，03:38:17.500开始；总180+TERM30（含加载与验证），165秒内部软停止留序列化时间。

固定采样seed20261024、temperature.6/top_p.95/top_k0、最多1024新token，原生BF16/SDPA、不量化，TF32初末false；不强制开启/关闭think，不重采样、不修答案。

| 固定input-only样本 | 输入token | 实际终点 |
|---|---:|---|
| first：10313222:relations:0 | 2171 | 1024新token，33.219182秒，30.82556 token/s含prefill；无EOS、token cap=true、wall cap=false |
| median：5789735:relations:48 | 3425 | 按首截断停规则未运行 |
| longest：6063015:relations:0 | 6002 | 按首截断停规则未运行 |

加载6.713929秒，实际399参数键/全部shape/BF16/finite、总8,190,735,360检查通过，无missing/unexpected/mismatched/errors；untied head保留。权重全SHA来自刚结束的资产收据，加载前后stat保持，小配置重hash；未把它说成再读16GB或逐元素文件/加载值比对。峰allocated19,495,760,896B、reserved19,505,610,752B。03:39:03.804实际卡空0MiB/目标无进程。

完整1024生成IDs及原始文本双份留本地和.100，row0 SHA `a7eaf566…`（20958B）；实际1个think开启、0个闭合。当前观测只证明2171输入这条、1024预算不足；不能当6002最长行/全部80稳定成本，也不能据截断宣称模型质量负。后继更大预算或全部80须由根线程据实测另定。

收据提取器最初硬编码启动前旧runner SHA而拒绝，原源/错误来源说明保留于[本地摘要异常](gpu-probe/summary-attempt1-stale-source/receipt.json)；只把本地常量更新为已经冻结的真实源/plan，重新读取已有收据，没有重模型/解析器/评分。

## 持久资源与恢复

持久mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，自有根目录：`/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/deepseek-r1-0528-qwen3-8b-readiness/`。模型在 `models/DeepSeek-R1-0528-Qwen3-8B-5ee5fa4832e6/`，CPU固定样本在 `cpu-preparation-v2/actual/`，GPU全部源/执行plan与原生输出在 `gpu-probe-v1/`；资产与GPU原job均终止，无待恢复运行。

本地完整证据：`R/research/deepseek-r1-0528-qwen3-8b-feasibility`（R=`/workspace/patientphex-runs/round2-20261023`）。Git只保存小源/收据/索引，完整提示、生成IDs/text、权重保留外部；每件SHA见合一清单和collection manifest。不得重发已有start脚本。
