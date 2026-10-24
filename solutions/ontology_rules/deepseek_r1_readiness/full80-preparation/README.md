# R1 同 F1 固定80对照准备

目标是准确R1对旧SFT32 .660942003 的关联增量。固定同80纯输入/16文献/587概念/39患者，不读新gold来设计；复用正确发布接口已冻row0/73，只有其余78生成。每行 fresh torch/manualCUDA seed20261024，不称连续RNG等价。发布tokenizer、独立派生scale1、BF16/SDPA、temperature .6/top_p .95/top_k0、4096 token及原codec均固定。

主线程已授权唯一10800秒+TERM30，含加载/保存，内部10770软停；上限不是预计耗时或所有行都完成的保证。每行cap/非法最终文本整行弃答，不重试/续写/补空；合法缺CID也是弃答，不能当明确负例。OOM/数值失败/总截止停止，partial不能评分成完整80。实际CPU部署核同正式load_inputs纯标准库路径；随后fresh .100 GPU1核UUID、0进程与空间才唯一启动。

候选按每row独立CID回译，原患者字符串保留。584个HP格式概念及3个原自由文本串均沿原code_map literal保留，不能另做HPO过滤/映射。独立[census](input-census-independent.json)证明跨row同文概念重复0、患者列表一致，1557可选patient-concept pairs。

固定primary为R1-alone关联集合；唯一secondary为旧SFT32集合与R1集合并。两完整预测共同冻结后，各唯一固定v2 all/text；旧metrics7e36dd93复用。每行association外字节保持oldprediction ffb2c54a精确。两个新臂都与强旧基线比较：关系micF1+macF1严格升且macF1不降才留用；不增加第三组合、阈值或自动迁F0/原dev/官方。

当前source/plan已冻结，未start；后继实际部署/CPU/作业写同目录。远端自有 `research/deepseek-r1-0528-qwen3-8b-readiness/full80-v1`，mount5a9955da-5240-41cc-9e52-05d3d8da8a3c，Python及源SHA见plan。UNKNOWN先查已发intent/原wrapper/job，不重新start。旧格式成本结果见[发布接口报告](../published-interface-v1/REPORT.md)。
