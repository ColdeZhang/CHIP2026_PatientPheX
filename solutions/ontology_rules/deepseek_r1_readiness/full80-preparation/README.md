# R1 同 F1 固定80对照准备

目标是准确R1对旧SFT32 .660942003 的关联增量。固定同80纯输入/16文献/587概念/39患者，不读新gold来设计；复用正确发布接口已冻row0/73，只有其余78生成。每行 fresh torch/manualCUDA seed20261024，不称连续RNG等价。发布tokenizer、独立派生scale1、BF16/SDPA、temperature .6/top_p .95/top_k0、4096 token及原codec均固定。

主线程已授权唯一10800秒+TERM30，含加载/保存，内部10770软停；上限不是预计耗时或所有行都完成的保证。每行cap/非法最终文本整行弃答，不重试/续写/补空；合法缺CID也是弃答，不能当明确负例。OOM/数值失败/总截止停止，partial不能评分成完整80。实际CPU部署核同正式load_inputs纯标准库路径；随后fresh .100 GPU1核UUID、0进程与空间才唯一启动。

候选按每row独立CID回译，原患者字符串保留。584个HP格式概念及3个原自由文本串均沿原code_map literal保留，不能另做HPO过滤/映射。独立[census](input-census-independent.json)证明跨row同文概念重复0、患者列表一致，1557可选patient-concept pairs。

固定primary为R1-alone关联集合；唯一secondary为旧SFT32集合与R1集合并。两完整预测共同冻结后，各唯一固定v2 all/text；旧metrics7e36dd93复用。每行association外字节保持oldprediction ffb2c54a精确。两个新臂都与强旧基线比较：关系micF1+macF1严格升且macF1不降才留用；不增加第三组合、阈值或自动迁F0/原dev/官方。

当前唯一原job `proc_12e6dfa0-4f21-4f0a-a2d2-4ad35a5ab248` 正在运行。实际纯CPU同入口预检0.084373秒通过；GPU main于04:12:06.254启动，主截止07:12:06.254+TERM30；fresh GPU1 UUIDfc5d/0MiB/无目标，416.9GB盘。实际源/plan 4a88bd33不变，见[入口清单](entry-manifest.json)。远端自有 `research/deepseek-r1-0528-qwen3-8b-readiness/full80-v1`，mount5a9955da-5240-41cc-9e52-05d3d8da8a3c，Python及源SHA见plan。UNKNOWN先查已发intent/原wrapper/job，不重新start。旧格式成本结果见[发布接口报告](../published-interface-v1/REPORT.md)。

评测入口纯合成检查通过。静态审查修正public evaluator按名导入validator的顺序：先用固定SHA加载canonical模块名，仍在双预测冻结后才import/读gold；旧未执行准备源12872737及template保存在evaluation-preparation-v1，新源4fbc58db，不是实际评分失败。

已知原运行I/O边界：每行正常/预期非法时完整raw/IDs已写；未预期parser异常则只有每32token partial，末不足32 token可能缺失。该问题在唯一job启动后才指出，主线程明确保持健康原job，不热改/重跑；遇此异常只能工程partial，不能进入完整80质量。后继版本应先独立raw落盘再解析，本run没有假称已修。

统一入口独审[b3670a88](entry-review.json)通过，未重NN/CPU。已准备本地单例finish_when_complete控制器：只观察原monitor，只有原作业完整终点才按已冻source执行回收→绑定动态SHA→两固定评测；每步intent，失败即停，不重模型/评分，不杀远端。STOP-AUTO-FINISH文件可阻止后继CPU步骤，原GPU预算独立保持。

04:29原只读monitor因RPC错误退出，原GPU不变；独立`monitor_recovery.py`接原同一lock/job，固定原observer07:17:36截止，正常60秒/错误冷却180秒，不发start/stop。原finish-controller PID101275仍等待，未重复启动、未触发任何评价。04:31:28实际33/80；这只是运行进度。
