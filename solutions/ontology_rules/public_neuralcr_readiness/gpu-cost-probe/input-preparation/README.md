# NCR GPU 成本探针：纯 CPU 输入准备

完成固定输入准备，未使用 MCP、GPU、NN、fit 或质量评测，未提交 Git。

- 先核冻结文件完整 SHA `1924f3691da766a81de963690d9a72bbd4f030ff34c3e191c082c64bc6ff7b96`，检查全部 408272 行严格排序、唯一及原生分词往返；0 失败。
- 原文件顺序零基 `[0,1536)` 原样选取，分成连续 3×512。没有跳过、重排、过滤、替换，也没有读取 gold/FN 或原始训练文献。
- `probe-inputs.json` 为紧凑 JSON，顶层 `batches` 是三个各含 512 条原样 query 的列表。完整 44193B，SHA `e4bda364b3c6011b813794db1a905834eba63d212e966506e1f6788735d267d4`，单文件小于 64KiB。三个独立 batch 文件也已保存。
- 纯 CPU 实际耗时 2.798753 秒、工具观测退出码 0；源码只 AST 提取固定上游 `is_number` 和 `tokenize`，未导入 TensorFlow/fastText。

`INT/FLOAT` 的确不能一般性地直接拼回文本再分词：`123→INT→int`，`nan/inf→FLOAT→float`。当前原生生成器先按非 ASCII 字母/空格切分，因此没有直接数字候选；仍需防 literal nan/inf。实际全 408272 行均无大写 `INT/FLOAT` 标记，且全部逐行满足 `tokenize(key)[:49] == key.split(' ')`，因此当前固定输入安全，无需修写。此前 census 仅保存 normalized keys，没有持久化 key→原始 query 的代表映射；四字段源输入的固定 SHA/路径仍在 census plan 中，可供必要时精确重枚举。本次未重枚举或读取原文。

完整统计和逐批 SHA 见 `result.json`，源/输入绑定见 `plan.json` 与 `handoff-manifest.json`。三批起始词面高度集中于词典排序的 a 前缀，符合确定前缀授权；不能据此声称覆盖全库内容分布。
