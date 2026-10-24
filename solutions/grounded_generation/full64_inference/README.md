# full64 完整训练后的原 dev36 推断

当前为后继源码草稿，没有实际 plan、checkpoint 绑定或 GPU 启动。只有原173×2/44更新完整结束、两份 .100 实际持久 ACK 与 campaign-complete 齐备后，才可绑定正式输入和最终 adapter SHA。

沿原 F0 runner 的同字节生成政策：完整输入、原 native backend、greedy1600、原生 tokenizer decode、先保存 suffix IDs，再保存原文、再本地解析。生成循环 AST 除进度39/47→36的条数元数据外与原 F0 相同；不修答案、重试 cap/JSON失败或从旧模型缓存冒充新输出。新 source 只将前置模型/完整状态身份改为 full64 的44步与两个持久ACK，输入取已实际209 CPU合同中的36条纯held。

推断预算1200+TERM30由root持续授权。必要规范化使用原 b59eb 阈值/公共41099银行，以及同算法的train64别名；不重新fit或编码公共bank。交付完整new-only与所有quote/精确跨度/拒绝原因，不读取原devgold、运行v2或上传官网。最终两个完整比较端点由root统筹。
