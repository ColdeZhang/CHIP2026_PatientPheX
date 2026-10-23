# 加载现有实体选择器

入口：[predict_existing_selector.py](../predict_existing_selector.py)。它只加载已保存的 HGB 与 DictVectorizer，按原特征函数计算概率、按原 greedy 解码；不拟合、不读训练/评测标签、不生成候选、不进行 Task2 推理。

## 调用契约

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python predict_existing_selector.py \
  --input /data/unannotated-input.jsonl \
  --candidates /artifacts/merged-candidates.jsonl \
  --base /artifacts/ontology-base-predictions.jsonl \
  --ontology-source /source/solutions/ontology_rules \
  --model /artifacts/selector.pkl \
  --model-manifest /artifacts/fitted-manifest.json \
  --cut 0.3 \
  --output-dir /artifacts/new-selector-output
```

使用可信的项目 checkpoint；pickle 不是不可信模型文件的安全解析格式。当前已保存模型需 Python **3.12.14** / sklearn **1.8.0**，入口核验版本、实际模型 SHA、特征表、类别顺序及全部特征模块 SHA。它不绑定原dev的文献ID、数量或输入SHA。

- `input`：无 `entities`/`association` 的原始文献 JSONL，保留 `pmc_id`、`full_text` 和 `patient`。所有文献ID唯一。
- `candidates`：逐文献 `{pmc_id, candidates:[...]}`，可以是现有 `selector-scores.jsonl`。每项保留坐标/原文/原子或复合HPO标识、原Sap `probability/similarity/exact`、base/CRF/source标记，以及独立 `public_*` / `bioformer_*` 六字段。应由已审 `multi.pool → merge_pool → merge_bioformer` 上游产生，不允许把两个模型分数合成一列。当前模型未使用CRF confidence，`crf_ner/crf_link`应缺省或0。已有 `selector_probability` **一律丢弃并重新计算**。
- `base`：相同文献的原本体预测，提供已有特征依赖的本体实体、同ID支持和上下文频数。它不是新的最终HGB结果。
- `cut`：有限的 `[0,1]` 数。当前已选定部署值为 `.3`；提供参数能力不意味着允许对dev重新扫门槛。
- 输出目录必须尚不存在。输出 `predictions.jsonl`、完整 `selector-scores.jsonl` 和版本/输入/来源/输出SHA manifest。默认输出只有Task1；空候选文献保留并输出空实体。

若需要装配独立Task2结果，额外传 `--associations /artifacts/relation-predictions.jsonl`。文献与指定患者键必须完整一致；每个 `association` 对象按原样复制，不生成、排序、规范化或按最终实体过滤。默认不会使用候选/base文件附带的关系。该参数不能自动把新HGB实体变成旧关系模型的候选池。

入口校验文献集合、重复候选、坐标/原文和唯一passage、有限数值、source字段类型、HPO标识语法及患者键。当前HPO分支与上游训练隔离仍由候选生成阶段保证；模型无法从坐标反推出候选来源。固定词表没见过的section/ptype类别按已拟合DictVectorizer原行为忽略，并列入manifest；不扩词表或重fit。

## 已完成的无评分回放

使用原冻结16篇input、13,144项合并池、实际295,637B模型 `db2fc288…` 和同cut `.3`；原关联 `aa107790…` 只作装配输入。实际重算约 **1.99秒**：完整预测 SHA **6401a24e…**、全部候选概率 SHA **c032ca68…**，均与已冻结结果逐字节一致。未读取gold或调用评测器，因此本次没有新增质量分数。

[check_existing_selector.py](../check_existing_selector.py) 完成16项合同：错坐标/文本/患者/缺source/NaN/重复ID拒绝；任意新文献ID和空候选文献接受；把旧概率全改成字符串后，实际一次 `predict_proba` 仍恢复原1,304项概率；同时设置所有fit入口和标注文件读取为抛错陷阱，均未触发。纯decoder函数AST与冻结校准入口完全一致。默认Task1-only没有association字段。测试只使用缓存原文和预测，不使用标签。

首次运行对余弦值机械限制≤1，误拒232项已存在的浮点结果（最大1.000001192）。现仅对余弦校验加 `1e-5` 浮点容差，**没有截断或修改任何特征值**；失败源码/trace与成功重放都保留。一次执行环境短断后先确认合同目录不存在，再补跑一次，没有重跑已成功的全16回放。

完整实际命令、源码快照、失败记录、合同检查和回放SHA在仓库外 `sequence_crf/selector-load-only/`；[小型证据索引](../evidence/round2/selector-load-only.json)。此入口完成已有候选池的实体推断，完整原文候选生成及单一端到端新输入流程仍需另行整合；未跑test100、未上传官方。

独立增量审已通过：`validation/selector-load-only-review/review.json` SHA `7c3b8401fbe769218606ca0fd6ba67c10f893e07415e40e1727f1091cd4ff149`。独立实际重算全16输出和全部概率byteexact，16项合同重放一致；没有读取gold、重fit或产生新质量分数。
