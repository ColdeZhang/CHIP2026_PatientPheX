# 病例检索研究笔记

只保存可复用的方法来源、工程经验和待验证假设；试验结果以[实验过程](experiments/E-20261023-008.md)为准。已确认跨方案坑回报主线程更新根 HANDOFF.MD。

首日结果见全局实验索引；第二轮不能只优化关系选择，现有实体/关系候选覆盖的诊断性上限不足 0.8，需要同时改善覆盖和患者归属。此上限是诊断，不是真实候选成绩。

## 本轮研究

- 只读研究在train64内部fold0诊断：1147个有效金标实体中704精确命中；443漏项有410项属于未见词面。建议将原文quote抽取和HPO归一化分开验证；完整研究记录在外部 `research-recommendation.md`，未读取dev金标。
- 当前旧RAG只能选择原候选代码，不能新增实体。新增抽取按文献/段落/窗口/阶段建立唯一row_id，缓存绑定prompt和源码hash，防止多窗口被同一pmc跳过。原文引用回定位不接受模型编造offset。
- 公开同赛实现 `https://github.com/yuanpinz/chip2026-PatientPheX` 值得参考逐occurrence、患者竞争与显式组患者关系；暂无明确许可，只参考思想，不复制代码。其自报严格LOO .68478与当前口径不同，不能直接比较。公开报告自由实体补漏未改善，故保留原候选关系判定先行对照。
- 全文讨论段落可能提供entity，但不能因此归属患者。实体输出和关系候选可分开校准；关系仍必须有原文证据，不凭疾病名臆造。

## 官方综述与后续对照

已读取官方综述 `https://arxiv.org/pdf/2610.29606` 的baseline说明：患者额外称谓/共指识别 → 每批8个表型、各出现±500字符语境判关系，Qwen3-8B greedy。当前新增 `prepare_coreference.py`/`merge_coreference.py` 和 `prepare_pair_rag.py`/`merge_pair_rag.py` 作独立对照；自写实现，非复制未许可代码。先跑最小全文关系基线，避免为准备所有可选模块推迟首个有效分数。

冠军公开LoRA配置可参考r16/alpha32/dropout.05与7个attention/MLP投影层，但其权重训练80篇包括我们dev16，禁止直接用于本地验证。若无监督Qwen收益不足，应只用当前训练折重建LoRA训练；原配置不等于收益证据。

新增quote归一化的候选可并用字符TF-IDF与冻结SapBERT CLS检索；训练别名仍只相应训练折，SapBERT来源沿首日已核验公开预训练权重，不新加人工表型数据。候选模型不会直接决定最终标签，未知/歧义项由Qwen在列出的候选中选取并显式记录无法解析项。

ModelScope对小文件完整Range可能返回200而非206；仅当请求严格覆盖全文件时可以接受200，必须另验长度和全SHA。部分范围仍必须206及准确Content-Range。分片任务失败应保留已下载文件并从缓存恢复，不重下16GB模型。

训练48抽取风格审计（未用于更改当前冻结zero-shot prompt）：3,612正实体中41含syndrome，INTRO+DISCUSS占48.75%，提示不能把“病例归属”边界套进实体识别。至少134正跨度不含紧邻严重程度/方位修饰，compound正跨度仅30；宽泛“complete specific phrase”可能损失精确边界，需要训练示例或SFT单独验证。305窗口有12个同词同时正/NO，去重quote→所有出现回定位不能完整表达逐出现断言差异。证据外存research-train48-style-notes与research-sft-style-diagnostics。

私有训练依赖在独立`training-runtime`安装peft0.15.2/bitsandbytes0.45.5，不修改已验证推理runtime；CPU隐藏GPU的import通过，不代表CUDA量化kernel已验证，仍须获GPU时做single-step。

原文quote输出即使逐字存在，也未必是正确实体边界：全substring回定位会把短词插入长词，广义临床抽取易加入严重程度修饰或基因/蛋白结构。归一化返回-1只能表示确实存在但未映射的表型，不能代替非表型删除；后续单独验证明确拒绝协议。跨窗口响应按row_id而非pmc_id读取，不能复用要求每文献唯一的read_jsonl。格式恢复只按截断/JSON结构判定选择，一次重试并保留原文件。


Qwen3.5-9B镜像来源须逐文件核验：固定MS commit权重/配置可等于固定HF commit，但LICENSE内容不同；保存HF原版并明确传输来源，不能声称两个仓库完全一致。模型合计9.653B仍须另计LoRA参数；新DeltaNet投影不能盲用旧Qwen3七target列表。

新A100服务器系统glibc2.17；同机某Python里torch2.5.1cu121导入会通过libcurand触发缺GLIBC2.27，而另一个torch2.6.0cu124 CPU导入成功。因此必须验证实际解释器/动态依赖链，不能按“版本更旧”推断兼容。只读复用成功环境、私有target更新必要包，不修改系统glibc或共享环境。CPU import通过仍不等于GPU/长上下文通过。


### Transformers 5 的加载报告类型

实际Qwen3.5加载报告中的missing/unexpected键集合使用set，直接json.dumps会在已加载权重后报错，尚未进入模型forward。证据编码器只将set/frozenset按repr排序成数组，保留成员与错误断言，不用str()吞未知类型；记录原失败，新输出目录重试。不能因保存报告失败就把模型接口判负，也不能因此跳过missing/mismatch/参数计数核验。来源见E-008及qwen35-loading-info-fix.json。

### 二层校准须隔离候选生成器

普通OOF行不能再次随意拆分作为meta训练/验证：生成其他训练行候选的模型可能已经见过meta验证文献。严格校准须在meta训练文献内重新生成OOF候选，vectorizer也只fit meta训练，不能为凑齐旧特征列数引入heldout词表。本次train32/meta16选定阈值.30，再固定作用原full48树，外层Score由.653192升至.669003，仍低同实体规则.711354；按停止条件关闭、不移原dev。单折校准有方差，外层此前参与过研究，不能把此结果写成盲测或把改善弱基线写成超过强基线。见[E-008的r2-18记录](experiments/E-20261023-008.md)及[小证据](evidence/round2/strict-meta-calibration-result.json)。

- 重复fresh运行环境检查应写新的launch-preflight文件，逐项比较解析后的模型/环境/SHA语义；不要覆写已有freeze所绑定的初次preflight，也不要把可能含耗时元数据的整份新响应SHA当作环境是否等价的判断。

- [关系连续置信度判断](research/continuous-relation-confidence.md)：现有Qwen raw无logprobs，不能直接调置信阈值；monoT5式yes/no logits是不同的连续信号，但4127训练pair实际prefix为4.145M（原6.39倍），收益/运行时间未证。患者CE、max-MIL、span-ID CE入树已做过的机制分别列明；当前只读保留，不追加GPU。

### 身份上下文覆盖不等于目标总分收益

同候选/同旧384前缀保留、尾部最多128 token原文身份锚点的严格fresh CE对照，训练579个缺目标marker正例中378获得新身份上下文，但外层Score .652285→.641438，低同实体规则.711354。关系净增8TP并少10FP让micro提升，患者macro下降仍使总分降低；不能仅凭结构覆盖、train loss或净TP宣布改善。结论只针对本次追加原句/CE/4ep/.5控制，原句可含临床事实，不声称单独身份token因果，也不泛化为所有身份模型无效。详见[固定配对结果](evidence/round2/patient-identity-bridge-paired-result.json)。

数值检查须兼顾正常恢复机制：FP16 GradScaler的非有限梯度可触发跳步/降scale，若在其step/update前直接fatal，会把可恢复溢出变为中断。原最长probe有限不足以覆盖真实shuffle batch；新诊断同顺序重现有限forward/2个梯度溢出，原生scaler确实skip且不动权重。双方共同改BF16 autocast、保留FP32参数/Adam和全部输入/监督后完成1032真实更新；这是透明数值修复，不是质量调参或对旧精度的质量比较。原失败、原因诊断与正式评分分开冻结。


### SFT概率/事实融合的证据边界

单meta-fold中，用fresh SFT32生成F1、cachedSFT48生成outer0，共同LR+上下文两HGB只±SFT四事实，Score .673525→.683810：相同控制下有正增量，与旧公开基座四事实负结果不同。仍低规则 .711354/直接SFT48 .695444，因此当前配置不迁移原dev。不能把这次16篇meta训练等同完整3折OOF，也不能把新共同LR特征加成归于SFT。原始结果与固定来源见[结果](evidence/round2/sft-meta1-result.json)。

固定候选补充关系的经验（2026-10-23）：旧候选缺失占F1 canonical关系FN154/209，单靠重训关系无法补齐。公开PubMed95∪Bioformer95全部新文档概念的机械追加在F1稍正、outer0稍负；只保留固定HGB cut.3文档实体支持的新关系后两训练折均正，再唯一原dev .732258→.736527（18TP17FP）。旧实体/关系不变，support在生成后过滤而非预删prompt。cut.3由pooled64实体OOF校准，不把F0/F1当阈值盲测；原dev已多次开发使用，仍需官方后继验证。见public-new-relations各结果证据，不用这些诊断逐条挑gold。

### padding下损失验证必须先保持同一shape

真实Qwen短行591与追加8masked pad的599在原生full CE本身出现−.018571差；不能用跨shape模型输出严格不变来证明/否定新loss mask。应逐shape比较原生full与选位loss，单独保存跨shape原生差；再在同冻结logits上poison被忽略目标，核loss和grad严格不变。实际probe通过后仍披露梯度容差，不称bitwise；失败日志原样保留。见value-weighted attempt2证据。

- Thinking后继：原full64 greedy1600首条未闭think是已完成负记录；随后旧SFT32首条官方sampling4096也触顶未闭think，已独审关闭。实际HF5.20.2无原生presence penalty，自定义需只减生成suffix已出现token一次、与multiplicative repetition分开；先固定单条格式/成本探针，不算质量。源码/预算边界见[一页提案](research/thinking-sampling-followup.md)。

- 等曝光不等于等optimizer调用：124样本/accum8每epoch16calls；直接翻倍248样本只31calls。顺序增强以四124逻辑epoch/O-O-O-O对O-R-O-R，原ID顺序与每轮短尾flush固定，496曝光/64calls。先CPU重放实际分组/LR，再谈等预算训练；不能把新四轮历史对比写成只改变排列。
