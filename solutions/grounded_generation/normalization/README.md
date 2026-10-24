# 精确原文跨度的独立标准化

目标是把生成器已经定位到原文的整段表型短语转换为官方 HPO 或有训练证据的 NIL。生成器输出患者的 `(unit_id, exact_quote, occurrence_index)`，不生成 HPO ID；本模块不替代患者归属判断，也不扫描全文生成旧候选。当前只完成 CPU 资产构建和接口检查，没有 held gold、质量评测、远端访问、新编码或新 fit。

`strict.py` 已冻结为 `31d380f582a6ab83e0c73bddbac983c5cf038b250dd2b29355c14f11d8eaebf5`。后继语义政策已在任何 Sap 查询分数或 held 模型输出前冻结为 [policy_v1.json](policy_v1.json)，SHA `b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20`；语义编码尚待批准队列中的唯一实际运行，当前不宣称新收益。

## 完整固定初版与恢复入口

[semantic.py](semantic.py) 的 `GroundedNormalizer(..., model_dir=..., bank_dir=..., device=...)` 提供 `normalize(quote)`、`normalize_many(quotes)` 与 `normalize_grounded(doc,quote,offset)`，严格别名之后只有 `reason="unmatched"` 可以调用公开 SapBERT。公共 bank 对所有 alias 按**概念 max cosine**聚合，要求 cosine ≥ .90 且第一、第二不同概念差 ≥ .03；这是一组事前保守工程边界，不称校准概率，也不做阈值搜索。已知歧义、NIL、大写短缩写、潜在并列复合、显式正常/否定、查询超过32 WordPiece，以及与赢家别名的方向/侧别/数字冲突均保持弃答。语义仅输出一个活跃 canonical HPO，不产生复合或 NIL。

可用 [normalize_quotes.py](normalize_quotes.py) 从 input-only 文档及包含 `pmc_id,quote,offset` 的 JSONL 批量得到三态结果；所有精确坐标先核对，原文输入不能含标注字段。输出还保存语义候选/分数证据与政策 SHA。公共 bank 与公开模型必须逐 SHA 通过，缺少资产时显式失败，不默默降为 strict-only。

[semantic-cpu-summary.json](evidence/semantic-cpu-summary.json) 实际检查全部87训练行、801目标词面（624 unique）：严格重代入为778 HPO、18 NIL、5歧义弃答；public-only 448接受/353弃答；逐文排除训练别名后488接受/313弃答。后两种各有2个公共映射与训练标注不一致，明确披露。语义守卫后 public-only 331次/283 unique、逐文排除289次/259 unique 可进入回退。它们是 train-only 控制，不能当F1泛化分数。12项新增语义反例合同通过，包括多个alias属于同概念、公共歧义同分、NIL不能被覆盖、无分数不能静默跳过。

唯一查询包已准备，尚未部署或启动。精确源码、输入、公共权重、旧 bank 和120+15秒绝对预算见 [query-plan.json](evidence/query-plan.json)。完整接手命令和队列见 [WORKLOG.md](WORKLOG.md)。由父 `/root/sequence_recover_0518` 接管：DocRE实际释放 `.3` 后只运行624个train-only查询并切片公共bank；缺失旧bank直接退出，禁止盲目重编码。查询后立即向root和span负责人交实际释放证据。

## 接口与政策

```python
from solutions.grounded_generation.normalization import HPO, StrictNormalizer

hpo = HPO(hpo_path)
entity_keys = hpo.entity_keys(entity["identifier"], entity["text"])
relation_keys = hpo.relation_keys(relation_value)
normalizer = StrictNormalizer(hpo_path, train32_path)
normalizer.assert_disjoint_input(held_input_path)
result = normalizer.normalize_grounded(doc, quote, global_offset)
```

- `HPO.identifiers(value)` 对**完整**合法 HPO 单编号/分号复合串做逐项 `alt_id → canonical`、去重和排序，允许分隔符旁空白。任一分量无效、废弃、分支外或混入 `-1` 则整串报错，不丢弃坏项后保留其余部分。
- `entity_keys` 返回 `("hpo", canonical_id)` 原子键；实体 `-1` 返回 `("nil", literal_text)`。`relation_keys` 只拆完整合法 HPO 串，自由短语即使含分号也保持整体；关系值 `-1` 报错，必须提供官方原文短语。例：`HP:0001250; HP:0001251` 拆两项，`low cholesterol; elevated triglycerides` 保持一个文本键。
- `normalize` 只匹配整跨度的公共 preferred name / EXACT synonym 或 train32 别名；不做子串剪裁、词干化、词序重排、标点删除或症状修改词删除。统一 casefold 与空白后的全部映射须一致，即使存在字面精确命中也不隐藏其他大小写冲突。
- 返回 `mapped`、`nil`、`abstain` 三态及候选证据；`identifier=None` 的弃答不会转换成 NIL。NIL 仅接受训练实体明确为 `-1`、无冲突且**原样字面**匹配的短语；患者关联应输出原始 `quote`，而非 `-1`。
- 公共短缩写的非字面大小写变体弃答。歧义不按频次或字典序决胜。复合 HPO 仅来自完整、有效的训练别名，不把一个模糊短语任意扩展成多个概念。
- `normalize_grounded` 验证全局字符偏移、Unicode 字符长度和唯一 passage 内逐字相等。父解析层仍须验证指定 unit 和 occurrence；本模块没有把 unit 称作自然句。

## 已完成的 CPU 证据

[strict-manifest.json](evidence/strict-manifest.json) 绑定 HPO `a5092cbd…` 和 F2+F3 train32 `38c353ea…`，与 F1 输入 16 篇交集 0。HPO 活跃分支 19,120 概念；公共条目为 19,119 名称与 23,207 EXACT 同义词。训练 2,362 实体中 2,352 能作为合法精确别名，其中 46 为 NIL；10 条无效/分支外 ID 逐条保存在 manifest，未转为 NIL。9 个 case/space 合并歧义见 [ambiguities.jsonl](evidence/ambiguities.jsonl)。这些是资产计数，不是泛化质量。

14 项有区分力的 CPU 合同通过：空白复合串、非法复合项、自然文本分号、NIL typed join、alt ID、废弃/分支外 ID、未匹配弃答、整跨度边界、大小写冲突、字面 NIL、Unicode 全局偏移、重复 passage、训练 SHA 守卫、公共 bank 不含 NIL/compound。运行：

```bash
GROUNDED_HPO=/path/to/hp.obo GROUNDED_TRAIN32=/path/to/train32.jsonl \
python -m unittest solutions.grounded_generation.normalization.test_contract -v
```

## 公开 SapBERT 与现有向量 bank

精确库存见 [asset-inventory.json](evidence/asset-inventory.json)；逐行复用证明见 [public-bank-manifest.json](evidence/public-bank-manifest.json)。本轮实际本地核了公开 FP32 原权重 `a4696930…` / 437,955,508 bytes，及只做 dtype 转换的 FP16 `dc5e5071…` / 218,990,904 bytes。固定 `cambridgeltl/SapBERT-from-PubMedBERT-fulltext@090663c3ae57bf35ffe4d0d468a2a88d03051a4d`；模型卡 Apache-2.0，来源为 PubMedBERT + UMLS2020AA 自对齐，未新增外部人工表型数据。safetensors 含 109,482,240 参数坐标和 512 个固定 position ID buffer，未加载模型推断。

本地可复用源码 `solutions/semantic_neural/run.py` SHA `b1005687…` 的 `Encoder`：纯公开 `AutoModel.eval()`，CLS → FP32 → L2，最大 32 tokens；`System` 在 bank 建立后拟合的旧 detector 不属于 encoder，也不在本路线复用范围内。历史运行环境为 Python 3.10.20 / torch 2.7.0+cu128 / transformers 4.50.0 / numpy 1.26.4 / tokenizers 0.21.4。当前本地默认 Python 没有 torch/transformers，未作模型加载成功声明。

已核旧 train64 bank 的 **42,575 条有序文本**与旧 bank SHA、cache signature 和 public encoder 一致。逐词按当前 HPO 重建的 **41,099** 公共词串全部能找到，剔除 1,476 条训练专有词串；新映射只取公共 HPO，有 1 条公共歧义。旧 `label_ids` 有 4 处被训练统计改成其他概念，已列明，**全部旧映射均丢弃**。因此可复用的是公开词串对应的冻结向量，不能加载旧 detector、概念频次、别名优先级或旧 `label_ids`。

截至本轮只读本地盘点，`/workspace/patientphex-runs` 的 `.npy` 清单未找到这份 SapBERT bank 实体；历史远端 locator 已保存在 inventory，未访问远端核它当前仍在。不能把此观察外推成所有服务器无向量。已有 `nested-outer1/inner0/manifest.json` 绑定同 train32 的 41,800 bank / detector17，但本地该目录没有序列化独立 SapBERT detector；`crf-work/pipeline.pkl` 是 CRF，不作替代。其他 train48/64 `reconstructed-detector.npz` 都不复用。

纯公共行准备、旧向量验证/切片和新公共编码均有独立可执行入口：

```bash
# 不读训练标签，不加载模型；输出可编码公共行及每行 HPO 来源（大产物放仓库外）。
python solutions/grounded_generation/normalization/public_bank.py prepare \
  --hpo /path/to/hp.obo --output-dir /persistent/new-public-bank

# 本轮已执行 audit，但未传 --bank；再传已取回的 bank 才实际逐 SHA/shape 验证并切片。
python solutions/grounded_generation/normalization/public_bank.py audit \
  --hpo /path/to/hp.obo --old-labels /path/to/bank-labels.json \
  --reconstruction /path/to/reconstruction.manifest.json \
  --semantic-source solutions/semantic_neural/run.py \
  --bank /path/to/bc3b715246322f5190fdaa5564ac161a8ef701617cdc268836fbbe94366910bb.npy \
  --output-dir /persistent/verified-public-subset

# 如旧 bank 不可取回，可用相同公共模型新编码。此命令尚未执行，不做任何 fit。
python solutions/grounded_generation/normalization/public_bank.py encode \
  --hpo /path/to/hp.obo --model-dir /path/to/verified-sapbert-fp16 \
  --device cuda --batch-size 128 --output-dir /persistent/new-public-bank
```

`public_bank.py` 仅准备/证明编码资产；后继 `policy_v1.json` 的唯一固定操作点与旧8B全文quote过滤器不同，不加载其 `.65/.92` detector。公开 bank 没有 NIL；低相似度只可弃答。
