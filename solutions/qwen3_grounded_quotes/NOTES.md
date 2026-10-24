# 调研与已知边界

1. 原Q3 strict32 closedCID/PID为124行/32update，F1并集+.002973但F0负；canonical-name/trie为87行/22update，新增52全FP。两者都不是原G逐字quote目标。copy/completion/DPO实际使用Qwen3.5-9B。有限检索未找到Q3原G87完整端点，不将此说成所有denseattention未试。
2. 原G targets-v2/train bf0091ef，F1.input c6426db4，全部messages/answer复用；train32 F2F3完全排F1/F0，不重造targets、训练aliases或监督。公开Q3 b968826d/MS26028140已缓存，8190735360+43646976=8234382336。
3. 复用Q3 native full64_component参数化train/common/state，canonical qwen3_entry已证明87/尾7适配；不要误用旧124硬编码train。旧canonical训练实际745.019s/phase751.674s、峰23617037824B；新quote监督更长，必须按新真实形状判断，不直接继承显存证书。
4. 当前唯一支持融合控制按root预锁，与独立seed同公式；不承诺此臂正，不修改旧负结果。普通G逐item parser/b59规范化保持；不偷用canonical trie或补JSON。

低优先后备：Sainz等EMNLP2021把关系改写NLI，作者Ask2Transformers Apache2 a3aed8f6；MoritzLaurer DeBERTa-v3-large-mnli-fever-anli-ling-wanli b3546ea6/MIT/435M，现成885242通用NLI pairs提供不同语义初始化。512窗口、患者同名/代词与max-window误报尚未解决，不另开实现或下载。原文 https://aclanthology.org/2021.emnlp-main.92/ 。

完整只读调研：仓库外 research/next-methods-20261024-1720/REPORT.md SHA6c6ace0666c42d18637f902dccc7c75dfc7aba81312e4e7a13cef81d43efb1bb，网络响应与项目来源SHA已保存。R-Drop低优先不立第三条；same-span ranker和DeBERTa-span旧准备已查重。
