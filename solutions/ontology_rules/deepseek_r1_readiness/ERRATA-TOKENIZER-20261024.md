# R1 tokenizer接口纠正（2026-10-24）

当前Transformers5.20.2 `AutoTokenizer.from_pretrained`按配置的LlamaTokenizerFast解析成实际 `LlamaTokenizer`，其已核源码重建BPE、normalizer=None、Metaspace预切分与Replace▁/ByteFallback解码。发布tokenizer.json却为ByteLevel。原检查只比较同AutoTokenizer的encode/apply_chat_template，不能证明它保留发布文件的完整编码语义。

唯一hidden-CUDA CPU诊断11.8335秒/rc0，原发布文件、旧80输入与已有2644生成IDs全SHA绑定：

- 80条rendered模板全部相同、词表与special IDs相同；Auto输入IDs对发布raw或generic均0/80相同。
- Auto encode→decode对80条原文0/80 exact，发布raw/generic分别80/80；ASCII/中文/换行/空格6小串为0/6 vs6/6。输入和输出两个方向均受影响，不能称纯解码修复。
- 已冻结2644生成IDs用发布`Tokenizer.from_file`和generic解码完全相同，合法JSON只含C003/C006两键；原strict完整8键政策仍拒绝。没有字符replace、补缺键、改原判定或新质量。
- 旧1024与4096 GPU的模型参数加载、时间、完整生成IDs、EOS/think状态、退出/释放等真实事实保留；它们使用的是TF5改写编码，不能当准确发布输入下的R1质量或成本保证。

完整结果`decoder-cpu-diagnostic/actual-result.json` SHA `0e7d10b5…`，真实Llama类/fileSHA `ccf940a4…`在同目录actual-tokenizer-source.json。5件完整output（含80行audit与解码文本）在外部D/decoder-cpu-diagnostic/recovered及.100自有namespace，SHA由result绑定。原CPU清单038d、1024终点d172、4096终点f8ab保持原bytes，统一审查以本纠正更新适用边界。

下一步必须先冻结发布tokenizer的实际输入ID谱系/长度并核generic与raw完整一致，不能沿用旧2171长度或称无输入变化；任何新GPU需按根线程后继计划。HF/vLLM缩放另有固定源码差异：native函数CPU4.195秒通过，64频率相同，HF原scale1.1386294361，vLLM及内存派生recognized attention_factor=1.0的scale1.0；原config不改，此仅数学函数合同，不能推断logits/格式/质量。

后继发布接口CPU已于新独立published-interface-v1/cpu完成7.9316秒：明确逐80 raw IDs==generic IDs==原生chat-template IDs，所有backend组件及往返/special一致。new first2014、longest5689，总249937；原收据median3131是upper-middle，常规中位数3129。只用于新独立pilot，不追改旧输入谱系。旧4096及接口纠正统一独审4cc5b614已通过。
