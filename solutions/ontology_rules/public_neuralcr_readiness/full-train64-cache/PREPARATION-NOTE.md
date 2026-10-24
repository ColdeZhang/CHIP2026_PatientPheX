# NCR 全缓存的 CPU 输入准备

`prepare_inputs.py --plan plan.json --output-dir .` 只生成输入查询和原模型列标签；本地只做源码编译与合成 AST 接口检查，完整 train64 枚举留在远端。没有读取 gold、模型加载、forward、拟合或质量评分。源码导入只有标准库定义，无目录扫描、环境修改或执行准备的副作用；CUDA 隐藏与 NumPy 导入只发生在显式 `run()` 中。

## 固定输入契约

plan 必须包含 `input`、`native_source`、`ontology` 三个 `{path,bytes,sha256}` 引用；可选 `source_sha256` 约束准备源码。`expected_keys` 必须为 rows=408272、bytes=16445155、SHA `1924f3691da766a81de963690d9a72bbd4f030ff34c3e191c082c64bc6ff7b96`；`expected_counts` 为 documents=64（接受同义 docs，但并存必须一致）、passages=2301、candidate_occurrences=683438。

从冻结原生源码抽取 `is_number`、`tokenize`、完整 `NCR.annotate_text`。枚举函数只复制 `matches = ... self.get_match(candidates, 1)` 前的原方法 AST，再返回 candidates/candidates_info；原前缀 AST SHA 必须是 `44a7eff872d5bf5d98466a887064c4533fbe57d1052c9ee5f35eda4d5caba0b9`。不导入整个上游模块，避免 TF/fastText 初始化。

逐 passage 使用原生 1–7 词枚举，核原始字符切片的规范化、坐标边界、跨度唯一性和 token 个数。对全部候选执行 `tokenize(' '.join(tokenize(query)[:49])) == tokenize(query)[:49]`，不修写 INT/FLOAT 大小写，不因失败另选词。最终唯一键严格字典序，逐行 `{"query":key}` 紧凑 UTF-8 JSON 加 LF，输出 `queries.jsonl` 必须与冻结 census 的完整大小/SHA 一致。排序前缀仅是确定顺序，不表示内容分布代表性；本准备不改变候选、阈值或重叠处理。

`columns.json` 是紧凑 UTF-8 JSON 加一个 LF：`{"labels":[原 ontology.concepts 的 14260 个 ID，最后 literal "None"],"root_id":原值}`。不按 HPO ID 排序，不用当前比赛 HPO 重排旧模型 head。要求原概念唯一、root 在其中。列顺序 SHA 为 `sha256('\n'.join(labels).encode('utf-8'))`，这里没有末尾换行；它与完整 columns 文件 SHA 含义不同。

NumPy 必须是 1.23.5；用两行合成数据核 dtype `[('column','<u2'),('score','<f4')]`，align=False、offsets 0/2、itemsize=6，并与 `struct.pack('<HfHf',...)` 原字节往返。这个小检查没有模型分数。

## 输出与后继接口

成功 `preparation.json.status` 是 `complete_input_only_preparation`。固定小型接口：

- `artifacts.queries`、`artifacts.columns`：各自 `{path,bytes,sha256}`。
- `head`：`concept_count`=14260、`label_count`=14261、`root_id`、`order_sha256`；完整 labels 列表仅在 columns.json。
- `counts`：`documents`、`passages`、`candidate_occurrences`、`unique_token_keys`；`pmc_ids` 按原输入文档顺序保存全部 64 ID。
- `refs`：`source`、`plan`、`input`、`native_source`、`ontology` 引用。
- `ast_sha256`：`enumeration_body`、`tokenize`、`annotate_text`。
- `numpy`：`version`、`dtype_descr`、`dtype_itemsize`、`dtype_offsets`、`packed_roundtrip`、`synthetic_rows`。
- `timing`：`total_seconds`、`enumeration_seconds`；另有开始/结束 UTC、精确字节与往返检查布尔值及无模型/标签/拟合/质量声明。

`extract_native(source_text)` 返回 `(enumerate_native, tokenize, annotate_text)`，后两方法相关的调用方式是 `enumerate_native(None,text)` 与 `annotate_text(adapter,text,threshold=0.8)`。完整 annotate 方法原样保留，后继 adapter 提供 `ont.root_id` 和查缓存的 `get_match(queries,count)`；准备阶段绝不调用该方法，不提供替代重叠算法。后继必须按自己的冻结 bound-plan 校验准备文件、查询和 head 字节，不能把此处准备成功当 NN 缓存已经完成。

目录可以已有源码、plan 和控制器；程序只拒绝覆盖自己的 queries/columns/preparation 及相应 `.tmp`。源或数据、NumPy、schema、AST、计数、坐标、token 或完整 SHA 错误均写 `failed_input_only_preparation` 小收据并退出 1，保留已有部分文件以便查证。已存在的 preparation 收据不会覆盖；这类重复运行拒绝及进程被 SIGKILL/磁盘写入失败由外层 controller 的退出/超时收据记录。任何失败都不进入 NN。
