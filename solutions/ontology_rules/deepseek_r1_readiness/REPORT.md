# DeepSeek-R1-0528-Qwen3-8B：准确检查点可行性

**当前资产终点：PASS。** 原唯一下载于03:36:58完成，1862.200秒，9文件/16,388,603,517B全部SHA与固定revision一致，399 BF16 tensors、8,190,735,360参数、两片连续payload与精确EOF通过；原managed已实际running=false。未重试/扩预算。actual完整收据5c7ed899、头ef4303dc；执行plan dc3ba5f1绑定既有CPU/YaRN，唯一短GPU尚未启动，无质量分。

官方overview §4.2 Agriv用Qwen3-8B与本检查点共同生成患者—表型证据，再以OOF/LightGBM融合。每个独立模型≤10B允许这一方向。其公开权重卡也列出准确模型；这支持研究价值，不证明本项目边际收益。现有本地记录没有找到该准确检查点的实际任务质量或缓存，普通Qwen3/9B思维模式负结果不能替代它的实测。搜索范围见existing-resource-search.json，未在可行性阶段新查远端磁盘。

魔搭 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` 固定revision `5ee5fa4832e658ab84298806de06ed2a167834bd`，MIT代码/权重许可。已取5个小文件并逐个对API SHA；仅HTTP Range读取两权重头共46,076B，**0 tensor payload**，核399 BF16 tensors、8,190,735,360参数，16,381,470,720数值字节与index完全一致。两完整权重预期SHA/大小在metadata/modelscope-files.json；该只读阶段尚未核权重数值字节；随后完整资产终点见顶部。config为36层、4096宽、32Q/8KV heads、128 head_dim、12288 MLP、151936词表、独立embedding/lm_head，固定YaRN配置原留。

**接口不能套普通Qwen。** 官方说明架构相同但必须用本repo全部配置；tokenizer_config为DeepSeek特殊token/LlamaTokenizerFast，chat_template没有enable_thinking开关。0528支持system，无需人为加`<think>`起始；不能用普通Qwen的enable_thinking=False宣称关闭思考。README的23K平均推理token描述大R1，不能冒称此8B成本。8B基准64K上限也不适合作赛事预算。

BF16权重约15.256GiB，理论batch1 KV为每token147456B，8192总token约1.125GiB；未含激活/logits/CUDA/碎片。A10040GB更有余量；A30 24GB在短上下文batch1可能可行，但仍须真实探针，131072上限不作为实际配置。现有A100环境记录torch2.6cu124/Transformers5.20.2，不代表本模型已经加载通过。当前A100 GPU2留给sequence，本准备不用GPU。

最小对照复用F1纯input80条/16篇/587CID（ddba03ff）及旧strict32 Qwen3.5-9B完整回答，不新增候选或看gold选难例。未来先exact tokenizer长度审计，再固定first/median/longest三条自然输出格式/成本探针：单180秒+TERM30、每条最多1024新token、seed20261024、temperature.6/top_p.95，无强制闭思维/重试；不完整或无合法final记缺失证据。失败不自动扩思考预算。只有成本可用且后续授权才做一次全部80条同池控制，先冻结DeepSeek-alone与预定old∪new再唯一评测。学习式融合需另备严格OOF训练侧特征，不能拿F1标签训练后又评F1；详细候选协议和原资源refs见feasibility-plan.json。

下一已获根线程授权仅资产准备：.100持久自有目录缓存优先，单2100秒+TERM30固定下载窗口、完整SHA/真实头核；不装环境、不加载模型、不GPU/新gold/评分。原metadata和当前源码先入Git，未知启动/监控先核原intent/job，partial保留，不重复下载。

资产唯一作业已于2026-10-24T03:05:56.359Z启动：`proc_1c596785-307e-4471-8541-c982943da8bb`，.100 mount5a9955da；绝对截止2026-10-24T03:40:49.472593+00:00+TERM30。原intent/job与远端启动标记是恢复依据；原Git401已由主线程03:07代推解决：4edacb9 push+lsremote exact，源/job已GitHub备份；远端3源SHA与Python3.10实际编译也通过。恢复入口`assets/poll_original.py`，不要重发start。

原生CPU终点已完成：首入口因TF5默认BatchEncoding与list比较失败4.791秒，实际单行input_ids逐项相等；只加return_dict=False后5.582秒通过全部80/16篇，消息/候选/患者原样。native长度852/3425/6002（min/median/max），固定index0/67/73输入2171/3425/6002；parser2正12负通过，未加载模型。YaRN实际64个FP32频率与HF4.51官方参考逐位/字节SHA相同，缩放均1.138629436111989；两版均不读取非标准attn_factor，保留原配置及warning，不推断其他引擎或作者意图。脚本首次占位符语法错误原留，未进CPU函数；修复后实际通过。完整[CPU清单](cpu-terminal-manifest.json)，资产仍在原35分钟窗口下载。

后继三条探针仅检格式/成本：若1024预算截断，它只说明此预算不可用，不否定R1全部质量机制；保存完整实际生成ID/token速度，是否值得更大预算或80条须由根线程据实测另定。

GPU探针5件源于03:25在.100实际全SHA及Python3.10编译通过，尚未启动。待原资产终点真实通过后才绑定执行plan；启动器保留唯一intent和远端marker，UNKNOWN先查原作业，fresh GPU1门检与180+30固定窗口不变。回收脚本只读取原进程与不可变终点，不因监控失败停止健康作业。

启动前窄修：主线程发现CPU/GPU rotary频率字节强等可能误拒正常舍入，实际TF5源证明FP32 buffer的meta搬运/原生重建路径。保存原源，改为明确拒绝未知dtype、同device原生64值与实际buffer比对，跨CPU/GPU记录差值，事前rtol=8×FP32 epsilon/atol=0；CPU对HF4.51原bitexact证据保持。新源796df0d7已实际SHA/编译通过，模型配置/权重/生成参数及唯一180+30预算均不变，没有额外GPU检查。
