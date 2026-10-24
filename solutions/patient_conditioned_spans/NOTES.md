# 机制与执行笔记

1. 同一患者×概念有多个跨度只产生一个袋标签，max 的梯度选择潜在证据；不把 nearest 或同概念所有跨度标为该患者的已知正证据。
2. 全文所有窗口通过 non-reentrant checkpoint 保留可重算图；重叠 token 可微平均，患者全部物理锚点各计一次。患者表示无 detach，关系损失能跨窗更新锚点 encoder。
3. patient-head 只见概念证据跨度的正负患者袋。解码必须同时 globalNER>0 与 patient>0。globalNER≥−2 的所有跨度及全部患者 logit 外部保留；不得以单患者头向全 L² 输出。
4. NO 跨度不是正 NER 或正袋证据；无可表示正证据的正袋 masked，而非负。3 个词片不精确正跨度的 covering 候选从负 NER 中排除。
5. 研究 census 的 2352 是有效概念实体记录（含 NO），本实现 2275 是全部非 NO 实体记录（含无效本体 ID）；口径不同不回写历史。814 个有效关系仍完整：813 可训练、1 masked。
6. 跨环境重新 gzip 只比较 decoded 字节身份，同时记录容器 SHA；同一个传输包仍逐字节核验（H63）。
7. 模型实际权重、未用 pooler/预训练 head、参数计数均须从加载实证核；禁止仅按模型名称猜资产或门槛。

事前修订：86b37baf 的 balanced-BCE 准备保留且未执行任何 GPU/质量。为保持本次关键变量为患者条件和 latent bag，正式 NER 改用原 GP 的 multilabel categorical loss；完整几何不变，新增高分稀有负例影响/有限梯度 CPU 合同。
