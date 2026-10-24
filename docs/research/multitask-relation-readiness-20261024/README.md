# 预训练联合抽取模型：新路线准备

目标是增加现有完整系统尚未获得的患者—表型关系或正确实体供给。当前原dev仍.754361398；本次只读调研没有质量结果、模型下载或GPU运行。

## 已核事实

- GLiNER2.5的公开boundary模型提供实体边界与关系头、schema描述及typed endpoint联合解码，区别现有仅用于实体span的GLiNER与从BiomedBERT新训的DocRE。库和模型发布元数据许可Apache-2.0。
- 固定[模型revision](https://huggingface.co/fastino/gliner2.5-base-v1/tree/54785d51df8d86a0d8f2eb212fec15871aad2c4d)，HF公开tensor元数据193,581,591参数；实际权重header尚待加载检查，不能只用模型名称作参数证明。配置`architecture=boundary`、`enable_relations=true`，max_len4096；encoder配置位置512但为relative attention，实际可用长度须由官方调用和输入验证，不擅自截断。
- 已找到[ModelScope镜像](https://modelscope.cn/models/fastino/gliner2.5-base-v1)。其774,366,564B权重SHA7274094d与固定HF revision的LFS identity相同；config和README SHA也与已下载HF正文匹配。tokenizer清单存在，实际tokenizer/encoder配置及权重还未下载校验。
- 固定[代码revision](https://github.com/fastino-ai/GLiNER2/tree/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c)，[JointIE接口](https://github.com/fastino-ai/GLiNER2/blob/ca4f68235b50fe4c3ecf3876851d748fe4b15c6c/tutorial/15-joint_ie.md)强调typed endpoints、方向及可行性。native候选/关系预算也是模型能力限制，需披露，不能无声丢文献或患者。
- 旧序列路线仅在较大GLiNER容量研究中记录“GLiNER2-large精确训练配方映射不明”；未找到该2.5 boundary/关系头实际运行。这里研究预训练联合抽取这一机制，不声称同训练数据的纯架构或容量比较。模型卡未披露完整预训练样本谱系，保持未知。

## 下一步边界

新分支`explore/schema-extraction`与独立worktree；初版先核官方库实际API、依赖隔离、患者输入坐标/重复提及与HPO规范化、完整文档分窗和边界恢复。权重优先ModelScope、绑定实际摘要。先完成可运行的公开预训练版本，再固定完整F1质量端点和GPU成本；零样本失败不泛化为监督适配无效。

初版不得读取F1标签构建schema、规范化词典或调阈值；所有候选/原始分数/被拒原因外存，关键来源与结果入路线台账。常规工程和短输入验证自主处理，GPU由root按真实释放统一排程；原有生成/患者训练、FLA及GP四折控制不让位于未准备好的模型。

详细来源与外部正文校验见[sources.json](sources.json)。GLiREL同样属于预训练关系模型，但其公开许可为CC BY-NC-SA4.0，本轮优先准备Apache许可且有可核ModelScope镜像的GLiNER2.5，尚未运行GLiREL。
