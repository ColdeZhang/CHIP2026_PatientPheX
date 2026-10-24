# 方法来源与适用边界

- 官方综述§4.2冠军使用Qwen3-8B、DeepSeek-R1-0528-Qwen3-8B与LightGBM/OOF，但没有公布LLM是否SFT/LoRA。此次为自有配方迁移控制，不称冠军复现。
- 原生Qwen3和R1均36层全注意力；新权重/原生模板变化必须披露，不能把结果单归因于attention。当前强normal是原124局部CID监督后追加全文的推断，不能换成后续全文训练而仍叫单基座控制。
- 使用训练completion监督位置的前一token hidden参与原lm_head，只省被ignore的logits；完整input/attention和原mean causal CE保持。数学、同实际logits以及目标runtime接口分别验证，不能把CPU通过冒称GPU已可训。
- 原文献隔离沿124行candidate generators16与base训练32核对，整个F0/F1排训练；新F1仅既有研发验证，不称盲测。大权重原缓存复用，首次GPU前单次完整SHA。
