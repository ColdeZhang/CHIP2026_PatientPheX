# 精确原文跨度的独立标准化

目标是把生成器已经定位到原文的整段表型短语转换为官方 HPO 或有训练证据的 NIL。生成器输出患者的 `(unit_id, exact_quote, occurrence_index)`，不生成 HPO ID；本模块不替代患者归属判断，也不扫描全文生成旧候选。当前只完成 CPU 资产构建和接口检查，没有 held gold、质量评测、远端访问、新编码或新 fit。

`strict.py` 已冻结为 `31d380f582a6ab83e0c73bddbac983c5cf038b250dd2b29355c14f11d8eaebf5`。父路线在完整 train32 目标 census 后决定唯一标准化政策；此处的严格政策是可运行初版，不能据此宣称生成器突破已有别名覆盖。

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

`public_bank.py` 仅准备/证明编码资产；语义 cosine 门槛、第一第二**概念**差距和 fallback 政策待父路线用 train32 证据冻结，不继承旧 8B 全文 quote+Sap `.65/.92` 过滤的失败结论或阈值。公开 bank 没有 NIL；低相似度只可弃答。
