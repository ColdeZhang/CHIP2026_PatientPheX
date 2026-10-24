# TF5 AutoTokenizer 与发布 JSON：一次原80文本对照

因 R1 发现真实家族重建差异，追加一次 hidden-CUDA 硬60秒CPU核。实际17.716秒，Med与Qwen3.5各80行，分别按现用原生模板与Qwen enable_thinking=False；无weights/gold/预测/评测。

两模型各80/80渲染文本，Auto encode与原tokenizers JSON IDs完全相同，模板tokenize也相同；双方IDs分别使用两种decoder、skip-special true/false均完全相同。总tokens Med238727、Qwen250696。此为固定文本输入范围证据，不能外推所有字符串/视觉音频任务。

Med GemmaTokenizer的normalizer/pretokenizer/postprocessor/decoder/model完全相同；added_tokens序列化不同但全部special IDs相同。Qwen3_5Tokenizer为BPE/ByteLevel家族；normalizer和postprocessor相同，pretokenizer/decoder/merges序列化不同，merges两方都247587项；Auto另有audio_start/audio_end/audio_pad三个special ID，发布JSON查词返回null。只据现有收据报告差异，不改旧模型或结果，也不将组件hash不等误报为本80文本编码失败。

完整逐行原件 complete.json SHA4ced43bacbc84b551a79d82704efad16144a447d1589e23051f367c6eb751e0c；小摘要 summary.json。原唯一proc_9504c546已停止。Med正式前只复用本结果，不新增模型/编码重试。
