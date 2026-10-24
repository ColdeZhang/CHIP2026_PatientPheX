# DeepSeek-R1-0528-Qwen3-8B：准确检查点可行性

官方overview §4.2 Agriv用Qwen3-8B与本检查点共同生成患者—表型证据，再以OOF/LightGBM融合。每个独立模型≤10B允许这一方向。其公开权重卡也列出准确模型；这支持研究价值，不证明本项目边际收益。现有本地记录没有找到该准确检查点的实际任务质量或缓存，普通Qwen3/9B思维模式负结果不能替代它的实测。搜索范围见existing-resource-search.json，未在可行性阶段新查远端磁盘。

魔搭 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` 固定revision `5ee5fa4832e658ab84298806de06ed2a167834bd`，MIT代码/权重许可。已取5个小文件并逐个对API SHA；仅HTTP Range读取两权重头共46,076B，**0 tensor payload**，核399 BF16 tensors、8,190,735,360参数，16,381,470,720数值字节与index完全一致。两完整权重预期SHA/大小在metadata/modelscope-files.json；尚未核权重数值字节，不宣称资产完整。config为36层、4096宽、32Q/8KV heads、128 head_dim、12288 MLP、151936词表、独立embedding/lm_head，固定YaRN配置原留。

**接口不能套普通Qwen。** 官方说明架构相同但必须用本repo全部配置；tokenizer_config为DeepSeek特殊token/LlamaTokenizerFast，chat_template没有enable_thinking开关。0528支持system，无需人为加`<think>`起始；不能用普通Qwen的enable_thinking=False宣称关闭思考。README的23K平均推理token描述大R1，不能冒称此8B成本。8B基准64K上限也不适合作赛事预算。

BF16权重约15.256GiB，理论batch1 KV为每token147456B，8192总token约1.125GiB；未含激活/logits/CUDA/碎片。A10040GB更有余量；A30 24GB在短上下文batch1可能可行，但仍须真实探针，131072上限不作为实际配置。现有A100环境记录torch2.6cu124/Transformers5.20.2，不代表本模型已经加载通过。当前A100 GPU2留给sequence，本准备不用GPU。

最小对照复用F1纯input80条/16篇/587CID（ddba03ff）及旧strict32 Qwen3.5-9B完整回答，不新增候选或看gold选难例。未来先exact tokenizer长度审计，再固定first/median/longest三条自然输出格式/成本探针：单180秒+TERM30、每条最多1024新token、seed20261024、temperature.6/top_p.95，无强制闭思维/重试；不完整或无合法final记缺失证据。失败不自动扩思考预算。只有成本可用且后续授权才做一次全部80条同池控制，先冻结DeepSeek-alone与预定old∪new再唯一评测。学习式融合需另备严格OOF训练侧特征，不能拿F1标签训练后又评F1；详细候选协议和原资源refs见feasibility-plan.json。

下一已获根线程授权仅资产准备：.100持久自有目录缓存优先，单2100秒+TERM30固定下载窗口、完整SHA/真实头核；不装环境、不加载模型、不GPU/新gold/评分。原metadata和当前源码先入Git，未知启动/监控先核原intent/job，partial保留，不重复下载。
