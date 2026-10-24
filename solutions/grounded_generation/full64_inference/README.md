# full64 完整训练后的原 dev36 推断

当前为后继源码草稿，没有实际 plan、checkpoint 绑定或 GPU 启动。只有原173×2/44更新完整结束、两份 .100 实际持久 ACK 与 campaign-complete 齐备后，才可绑定正式输入和最终 adapter SHA。

沿原 F0 runner 的同字节生成政策：完整输入、原 native backend、greedy1600、原生 tokenizer decode、先保存 suffix IDs，再保存原文、再本地解析。生成循环 AST 除进度39/47→36的条数元数据外与原 F0 相同；不修答案、重试 cap/JSON失败或从旧模型缓存冒充新输出。新 source 只将前置模型/完整状态身份改为 full64 的44步与两个持久ACK，输入取已实际209 CPU合同中的36条纯held。

推断预算1200+TERM30由root持续授权。必要规范化使用原 b59eb 阈值/公共41099银行，以及同算法的train64别名；不重新fit或编码公共bank。交付完整new-only与所有quote/精确跨度/拒绝原因，不读取原devgold、运行v2或上传官网。最终两个完整比较端点由root统筹。

下游必要Sap待当前A800任务完成后排.100空闲卡：公共bank/labels/manifest由span路线已同字节复制bc4ea231/60a27430/443922da，不重编码。原.3 Torch2.7.0+cu128/TF4.50.0与目标.100 Torch2.7.1+cu126/TF4.51.3不同，后续实际plan必须绑定真实环境/相同6个MODEL_FILES与固定CLS_L2/FP16/max32；当前旧held草稿仍指.3，实际启动前完成机械迁移与真实资产身份绑定，不能直接运行该草稿。

最终最小补包入口为 `persistence/full64_supplement.py`，仅在完整纯源和集中终审齐备后执行。原终点ZIP逐字节收进包；两完整checkpoint只引用实际.100持久ACK，不把SHM冒称持久，不复制旧F1/F0包或公共权重。当前仅语法/静态路径核过，尚无本轮补包。
