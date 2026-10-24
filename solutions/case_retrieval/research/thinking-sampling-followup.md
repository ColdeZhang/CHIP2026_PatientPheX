# Thinking 后继：只做单条可用性检查的提案

**实际终点：格式失败，已关闭，不扩大。** 原SFT32/固定首条true2030token，官方采样加真实generated-only presence，唯一生成4096token/186.896秒、全native209.493秒，触顶无EOS、未闭think、没有合法final；peak18.725GB，01:03:39.439Z A800空释放。presence实际调用4096次，9个CPU反例先通过。完整freeze后只调用strict-final，不读targets/Score/拟合；独审 `e5f68aef…` 通过的是执行及失败结论，不是质量。环境恢复后只续接已存在job的日志cursor和原采集合同，没有重启NN；小证据见[结果](../evidence/round2/thinking-sampling-result.json)及[独审](../thinking_sampling_followup/independent-review.json)。

只读通读完整输出：仍围绕P1与8CID，旧定性摘要对“前半各项中间判断”的表述过强：真实generated-ID重放确认，7项明确Result列表结束于token1721，C003首次肯定映射结束于2214、首次明确列表结束于3529，全8连续复核结束于3704；后段继续讨论C003语义/亲属归属/是否足够明确。最后200token又复核否定项和C006，再问C003能否排除，停在“Sometimes”；没有进入最终JSON。不是无关话题或逐字固定循环，而是任务内重复复核；不能据此推断增加少量预算必能成功。原输出和旧诊断冻结不改、无标签正确性裁决；[旧诊断](../thinking_sampling_followup/behavior-diagnostic.json)的事件覆盖表述以[逐token只读更正](../thinking_sampling_followup/coverage-correction/REPORT.md)为准。


**以下为运行前只读提案，后继终点见本节。** 已完成的 thinking 控制不是空白：full64 epoch2、原有限 CID/患者提示、greedy、1600 输出上限，第一条实际 72.048 秒后触顶，未闭合 `</think>`、无合法 final；只完成 1/12 并早停，不读 targets、不报拟合。独审 `67b49e51` 保留。尚未做官方建议 sampling 或更长输出上限的真实探针；本提案不能把旧失败改写成成功。

## 参数是否真实生效

固定官方模型卡（README SHA `c5f5a8c2…`，832行附近）推荐 general thinking：temperature=1、top_p=.95、top_k=20、min_p=0、presence_penalty=1.5、repetition_penalty=1。其同时明确不同框架的支持不同。

实际私有 Transformers 5.20.2 的 generation 配置、processor 和 generate 源码都没有原生 `presence_penalty`。**不能把 `repetition_penalty=1.5` 当替代，也不能仅传一个同名 kwargs 就声称生效。** 当前 HF 支持自定义 `logits_processor`；实际源码先合入自定义 processor，再施加 temperature/top-k/top-p。若后续执行，最小实现应只对本次生成 suffix 中已出现的 token 减一次 1.5：`score[v] -= 1.5 * 1[v in generated_suffix]`，不计 prompt，不随次数累加。这个定义与公开 vLLM 的 OpenAI presence 实现一致；其 repetition 另作用于 prompt/output 且是乘除惩罚。

实现前必须有 CPU 反例：prompt-only token 不减、生成一次/重复多次均只减一次、正负 logit 都减同一量、两 batch 不串、空 suffix 不变。明确 prompt 长度边界包含 chat template 的 open-think，惩罚覆盖实际生成的思考与答案。没有新约束解码、禁止 token 列表或额外 system 指令；它会同时作用于 JSON/CID 重复，这是该官方配方本身的代价，不因本任务偷偷豁免。本页只证明接口/数学可行，尚未实现或验证 GPU sampler。

## 唯一候选探针与预算

| 项目 | 预先限定 |
|---|---|
| 输入 | 原 `meta.prompts.jsonl` 第一条 `10313222:relations:0`，按文件顺序选择，8 CID/1患者；只复制 system/user，绝不上传 target。原非thinking实际2032 input tokens；true模板须在启动前重新实分词，禁止截断。 |
| 模型 | 已训练 strict train32 JSON SFT32 epoch2 `4f4815b6…`，原公开 base/原 tokenizer；不是 weighted/full64 adapter。 |
| 生成 | enable_thinking=True，do_sample=True，seed=20261023，temperature1/top_p.95/top_k20/min_p0/repetition1，明确自定义 generated-only presence1.5；max_new_tokens4096。 |
| 格式 | 复用已验证 strict-final 契约：prompt已开think时必须实际输出闭合 `</think>`，只解析其后的完整 JSON；CID 全集/合法患者、无重复键、无触顶截断。思考段中的 JSON 不算答案。 |
| 终点 | 只记录实际 tokens、闭合/strict final、EOS/触顶、耗时/显存；不读 labels，不算 Score/拟合，不做第二 seed 或重试。闭合成功仅证明该条可用。 |
| 拟议资源 | 等 root 的 kernel/资源安排；建议一张 fresh A800 的单次总300秒+TERM30，含加载/校验/生成，状态保留且超时不延长；当前没有运行授权。 |

旧 A800 真实 1600/72.048 秒只给出约22.2 token/s的局部参照，4096线性代理184.44秒，另需加载/哈希/首prefill；更长序列、sampling、不同输入与kernel均可改变耗时，不能承诺300秒内生成完整。当前 A100 weighted80 的513.989秒不是该单样本吞吐测量，不混算。若 kernel 导入/数值检查改变部署或成本，应先固定实际入口后再决定是否执行。

它同时改变旧失败的 adapter/样本、采样、长度，因此是**单样本格式和成本诊断**，不是证明 thinking 质量提升的单变量实验；无论结果如何，都不能据此直接迁移原 dev。

证据：原失败 [qwen35-thinking12-result.json](../evidence/round2/qwen35-thinking12-result.json)；本次实际 HF 源码/公开惩罚快照和固定输入指针见 [thinking-sampling-readonly.json](../evidence/round2/thinking-sampling-readonly.json)。官方模型卡：[固定revision](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md)；presence 参考：[vLLM apply_penalties](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/utils.py)，本次下载字节另存 SHA，未安装或替换框架。
