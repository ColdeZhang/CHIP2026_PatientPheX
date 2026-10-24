# 来源与适用边界

- DPO作者论文：[2305.18290](https://arxiv.org/abs/2305.18290)；作者代码固定 `f8b8c0f49dc92a430bae41585f9d467d3618fe2f`，`trainers.py` 45–96原sigmoid、99–129 shifted/summed log-probability，Apache-2.0。原始网页、源码、许可证的实际下载SHA在仓外 `public/fetch.json`。
- Dropout策略也符合成熟实现的公开默认：TRL `v0.25.1` 注释tag解引用commit `1cf8d56c06ce095aeaa92d0853fca68e5834a928` 的 [DPOConfig](https://github.com/huggingface/trl/blob/1cf8d56c06ce095aeaa92d0853fca68e5834a928/trl/trainer/dpo_config.py#L306) 将`disable_dropout=True`设为默认，用于policy/reference。实际下载27223B/SHA `9ee8be6a674255e23bd9dab9e75d16aaab18a3b39ea0ce6924ebbe43a2a885fc`；本项目没有导入TRL，而是明确把原LoRA .05执行dropout置0。此来源补证不改变已冻结配方。
- 已试机制地图：主分支 `docs/research/method-coverage-map-20261024/MAP.md`。按监督/输入/目标实际查重，不能把BCE改名为DPO。
- 原Qwen3父与runtime：`explore/qwen3-relations@c7326308`，`solutions/qwen3_relations/{plan,f0-plan,f0-export-plan}.json`。仅复用train32 epoch2，不能误拿full64 `59fb484a` 适配器。
- DPO不增加缺失HPO概念供给，也没有显式删除背景文献表型的负例；它只测试同文多患者误归属这个受限假设。17篇对照数据较小，不能用SFT loss或合成偏好准确率代替完整held Score。
- 相同关系条数不保证相同token数；2对−1/2对+1已披露。固定sum符合所选DPO，不按新结果改成长度平均。
- Serial backward的链式法则正确不证明GPU no-grad/grad执行完全相同。源中先保存observations，任何实际非finite/显著差异须停并保留原件，不静默放宽容差。
- **实际混合精度缓存坑**：同一个autocast scope中先no-grad再grad，前向分数可以完全相同，但checkpoint backward仍会发生saved/recomputed metadata错位。首次大GPU已观测此错误；tiny CPU同BF16/nonreentrant checkpoint只复现梯度存在/finite门失败，未复现同一CUDA异常。关闭`cache_enabled`后CPU28梯度/Adam及实际GPU13对/26FB/504Adam状态通过；只固定缓存行为，保留BF16和原replay容差，不能把CPU FP32数学合同当完整执行证明。原source/plan/失败均保留，见实验记录。
