# 方法与证据

方向来源为 GENRE 的 canonical entity name + prefix trie；REBEL 提供集合序列化参考。公开来源固定版本与许可、旧开放 HP ID 配方的失败终点以及当前 grounded exact-quote 的机制区别，统一见主线 `docs/research/canonical-name-generation-readiness-20261024/`（原始准备另存外部）。代码为自写的小型严格名称接口，不复制 REBEL 的宽松 parser。

标准名称合法不代表临床含义或患者归属正确。生成端仅允许公开 specific phenotype 名称；不使用训练频率筛 held 名称，不对未知输出 fuzzy repair，不从 gold 修答案。标准名目标从 association 的合法 HPO 生成，不依赖 gold entity、nearest mention 或 nearest quote。输入给定的患者 mentions 仍用于身份定位。

H38 要求完整 prompt+assistant+EOS 的原生分词，并核真实公共前缀；不能把名称块的单独分词自洽替代真实模板连接。H48 要求 native token IDs 与实际运行对象比较。H49 要求解析前持久化 raw IDs／raw text／EOS/cap；未知、重复、触顶整行失败显式记录。

readiness 曾把 preparation-plan 的历史 schema SHA 当作当前 guard；已记录失败并冻结实际源码 460dd7cc，终态通过。这里消费冻结快照，不重演该错误。
