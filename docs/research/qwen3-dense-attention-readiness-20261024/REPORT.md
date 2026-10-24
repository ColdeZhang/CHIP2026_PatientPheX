# 原生 Qwen3-8B 关系控制：有界查重与准备结论

**值得补一次严格 train32 关系基座控制，但不是首次全注意力，也不是冠军复现。** 现有62个非审计训练manifest中，8个标准Qwen3匹配（含重复/工程）全部train48；未找到同F2+F3 train32关系SFT＋完整F1。旧Task1抽取确有train48→F1，不能声称该模型从未在F1运行。冠军综述只确认两LLM＋LightGBM/OOF，没有披露LLM是否SFT或LoRA。

## 历史与可归因边界

详见history/REPORT.md、result.json及sources.json。原生Qwen/Qwen3-8B固定HF revision `b968826d9c46dd6066d109eabc6255188de91218`，与已测DeepSeek-R1-0528-Qwen3-8B是不同发布权重。R1也为全注意力，因此新增变量是不同instruction/蒸馏前后基座及其必要原生模板，不是attention单因素。

| 历史控制 | 实际范围与终点 |
|---|---|
| 旧CID关系SFT | train48→F0，204行2轮52更新；A800训练599.857s/峰29.483GB，58条推断172.034s；Score .674333431，414/415严格解析＋1处既定LR回退 |
| 旧开放患者集合 | train48→F0，126行2轮32更新；训练654.412s/峰42.096GB，adapter raw Score .397872；不同于当前闭集PID→CID |
| 旧Task1抽取F1 | train48含F0→F1，305行2轮78更新；A30训练542.038s/20.565GB，完整88抽取＋23重试＋56规范化；filtered.rules .643815低于base .644796 |

## 公开资产与实际缓存

固定ModelScope revision `26028140be3ee69b82b1d1450179ab71bb1121b9` 的13文件API条目，大小/SHA均与既有已核缓存清单一致；本次官方HF config、README、LICENSE及ModelScope config/API均实际HTTP200，只取小元数据，无权重下载。Apache-2.0许可实际LICENSE SHA832dd9e0。config SHA f7c4eadf；36层、4096hidden、12288FFN、32Q/8KV、head128、无sliding window、无MoE/linear层、untied embedding/lm_head。模型卡native context32768；config40960不擅自当已测上下文上限。

实际只读safetensors头核399张量、全BF16、index映射与shape一致，参数 **8,190,735,360**；r16全部7线性层另43,646,976，总 **8,234,382,336 < 10B**，按每独立模型规则，不能用全流水线合计否决。13发布文件16,397,443,036字节（约15.27GiB）；目录附属本地文件不算模型权重。

- `.100`当前路径 `/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/round2/models/Qwen3-8B`，实际存在；磁盘空余409,082,859,520B。本次核小文件SHA及5个权重header/shape，未重读16GB权重主体；首次GPU使用前应单次核完整权重SHA。
- A800当前 `ontology-round2/pretrained-qwen3-8b`亦存在同399张量，盘仅1,483,972,608B，不足舒适保存双epoch完整状态。旧全SHA转移报告可追溯；当前header不冒称最新全body SHA。
- `.3`历史 `shared-models/qwen3-8b-b968826` 13文件已逐SHA转移，但本次按root离线恢复协调未再查询该mount。A100此基座副本未本次确认，不假定现成。

## 已完成真实CPU兼容与成本判断

`.100`现有TF4.51.3/Qwen2TokenizerFast完成唯一CPU census **11.979837s**，CUDA隐藏、未加载权重：124训练行及80全文F1的Auto/native/raw tokenizer IDs与双解码逐条全等，六个发布backend组件全等；训练prefix边界和JSON答案＋原生EOS＋换行全等，无截断。源分别cb7f67e1（原124闭集监督）与2a95bad1（原normal全文F1输入）。已知训练targets只用于分词；未读新gold/fit/评分。

- 训练：32篇124行，总403,151 tokens，最长5,534（11235872:relations:8），中位3,323，监督7,170 tokens；保留原2轮/32Adam、r16、标准mean CE。
- 全文F1：16篇80行，总666,591 tokens，最长14,322（12455887:relations:40），中位8,399。不是后续max13549的全文训练配方，不换监督内容。
- 单纯按旧原生训练峰29.48GB不能假定24GB能训。indexed-head只计算有监督位置的lm_head＋标准shifted mean CE可省无效大词表logits，不截输入；其数学/实际runtime合同及已有Adam/累积梯度最长压力须先测。
- 历史数量级估计2轮350–900s、80全文推断250–900s，非承诺；新基座/全报告token与设备不同，须真实短探针后校准。14k长prompt的KV缓存约2GiB（36层×2KV×8heads×128×2bytes×14322），加输出与激活另算。
- 两epoch每份adapter约174.66MB＋训练state约349.62MB，合计下界约1.05GB，原缓存外建议至少3GB余量；.100足够，A800需另协调。无需先下载16.4GB。

## 建议的最小完整控制与后续授权状态

固定原124行normal闭集训练2轮、标准CE、原生nonthinking模板；完整80全文输入与原schema/parser政策保持，不新增示例/阈值或学习融合。预写两个端点：新单臂，以及历史完整fullsupport(.686338981)与新预测的整体并集；两完整pred共同冻结后各一次v2/旧0，至少一项严格高于fullsupport才有F0投入依据。所有source排F1，既有研发/方法选择披露为开发验证。

root已于本调研末转交新branch `explore/qwen3-relations` / worktree `explore-qwen3-relations`，base ff676a0，独立新线仅solutions/qwen3_relations；DocRE保持负结案。后继获授权CPU＋420+30真实工程→通过后fresh2ep2400+30→80fresh1200+30，但**本报告时尚无GPU/fit/质量启动**，只是已完成读查和token census。若24GB不容纳，保原失败等待A800调度，不缩输入或盲重启。

## 记录限制

首次库存工具请求误用`timeout`，schema在执行前拒绝；改为已发现的`timeout_seconds`后两台各执行一次实际只读检查。首次本地网络脚本缺requests，未发请求；改stdlib urllib后五份小资源成功。错误记录保留；均无模型/数据副作用。全部文件在本外部目录，主报告由root收录，无Git大产物。
