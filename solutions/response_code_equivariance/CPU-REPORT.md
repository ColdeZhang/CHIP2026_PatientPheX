# cyclic1 CPU 准备结果

完整80行通过，固定一次响应PT循环移位：53条需新NN，27条单患者原JSONL、messages、nativeIDs完全不变，可复用原raw。完整16篇/39患者/587CID/1557对、clinical原文/markers、原患者顺序不变。结构17合同通过，不训练、不读heldgold、不评分。

唯一.100 CPU proc156e646f于13:30:20.2509实际结束rc0/11.7197秒，native10.9888秒。原/新总tokens均685372、最长14570/idx19，guard14848零截断；53新NN共475349tokens，27缓存210023tokens。全部Auto/nativeJSON/template及decode exact，53条ID确实变化。

实际Torch2.7.1+cu126、Transformers5.20.2、PEFT0.21.2、tokenizers0.23.1，Qwen3_5Tokenizer；实际Qwen源码b4b17及generation7944 SHA核过。后续如果使用.100 A30，原C32是A100Torch2.6，因此这是映射加host/runtime的开发控制，不能把差值纯归因编号。缓存仅复用已存在原输出，不声称跨runtime重新生成逐位等价。

未来固定两端点仍主B4∪(原Cstrict∩新Cstrict)、诊断新Calone，完整共冻再各一次v2；主无增量继承原指标不重评。主严格胜B4才讨论同公式F0；诊断独自更高也如实报告root候选，保持身份且不自动迁移。没有第三臂、setdecoder修复或扫描。GPU/质量尚未授权或执行。

单一恢复入口外部CPU-ready.json；全部8件真实CPU终点zipdc65c79a已SHA回收，源、计划、实际收据均独立保留。
