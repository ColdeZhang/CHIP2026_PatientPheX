# 来源与适用边界

- DPO作者论文：[2305.18290](https://arxiv.org/abs/2305.18290)；作者代码固定 `f8b8c0f49dc92a430bae41585f9d467d3618fe2f`，`trainers.py` 45–96原sigmoid、99–129 shifted/summed log-probability，Apache-2.0。原始网页、源码、许可证的实际下载SHA在仓外 `public/fetch.json`。
- 已试机制地图：主分支 `docs/research/method-coverage-map-20261024/MAP.md`。按监督/输入/目标实际查重，不能把BCE改名为DPO。
- 原Qwen3父与runtime：`explore/qwen3-relations@c7326308`，`solutions/qwen3_relations/{plan,f0-plan,f0-export-plan}.json`。仅复用train32 epoch2，不能误拿full64 `59fb484a` 适配器。
- DPO不增加缺失HPO概念供给，也没有显式删除背景文献表型的负例；它只测试同文多患者误归属这个受限假设。17篇对照数据较小，不能用SFT loss或合成偏好准确率代替完整held Score。
- 相同关系条数不保证相同token数；2对−1/2对+1已披露。固定sum符合所选DPO，不按新结果改成长度平均。
- Serial backward的链式法则正确不证明GPU no-grad/grad执行完全相同。源中先保存observations，任何实际非finite/显著差异须停并保留原件，不静默放宽容差。
