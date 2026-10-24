# 方法与复用

原 `response_codec.parse_matrix` 在完整JSON、每个CID齐全、list元素为字符串之后，检查患者code是否重复，重复即整行拒绝。新codec沿用原 `unique_object` 与原响应身份映射合同，先核所有PT合法，再按每CID独立稳定去重。重复的未知值仍拒绝，不因为去重隐藏非法内容；不同CID里的相同PT保持独立。

只把集合冗余当作允许的新解码语义，不重写原响应文本，不再序列化文本交给旧parser，不补字段或猜标签。原生成 provenance=fresh 仍指旧模型当时的生成，本次全部为已有raw的CPU重解，0新生成。新canonical的通过标记按新集合政策解释；audit保存原严格状态与原记录。

原coded-alone使用所有有效row的CID→patient映射，以 `row.code_map` 精确映射HPO并按文档患者取集合。新实现复用原renderer；旧80 canonical先重建原预测且须逐字节相同，再输出新coded-alone。不额外融合强N，不扫描任何子集。

## 本次证据

稳定去重只折叠了1个列表冗余，却使原整行的5个关系进入完整预测；它们在唯一F1下均为FP。集合语义上的可解析性并不证明原弃权行的关系正确。原strict解码继续保留；本新政策负结案，不因结果再筛选该行或CID。
