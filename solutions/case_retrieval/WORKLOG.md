# 病例检索工作台账

实验 `E-20261023-008`；本轮从合并后的 main 建立独立分支。

- **目标**：固定 train64/dev16 与公共近似评测 v2 下，Score ≥ 0.8；保持训练/验证文献隔离。
- **状态**：进行中；固定基点 `00f7da8`，训练标签复合值修复及 fold0 对照准备，Qwen3-8B 正在远端下载，私有推理环境已就绪。首日本地 0.613517597，官方 0.599432522，两者不可混比。
- **记录入口**：[实验过程](experiments/E-20261023-008.md)、[研究笔记](NOTES.md)、[小型证据](evidence/round2/README.md)。
- **运行位置**：外部 `/workspace/patientphex-runs/round2-20261023/case_retrieval/`，完整日志、预测、模型不得入 Git。
- **当前结果**：内部fold0原版 .613151 → compound修复 .615159；实体不变、关联微升，5项行为测试通过。固定 dev16 已得到 .625073738（原 .613517597），仍未达标；独立复评、重新拟合逐字节复现、36组内部输出复评全部通过。
- **下一步**：Qwen到位先原候选关系判定，再独立对照原文表型抽取与HPO检索归一化；完整准备与进程信息见实验记录。
- **边界**：只修改本方案目录；共享变更与主线程协调；本轮不得上传官方。关键检查点及时 commit/push，回报版本与证据。

## 运行与恢复

- Qwen下载：10.5.2.100，mount `5a9955da-5240-41cc-9e52-05d3d8da8a3c`，`case-retrieval/round2/`；进程 `proc_e6acd215-04c3-4006-9bcf-a80abf9365fc`，完成以 `models/Qwen3-8B/download-complete.json` 及全部源SHA为准。
- 本路线仅用GPU1 `GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539`；GPU0由主线程分给本体路线，共用只读Qwen文件，各自输出目录。
- 解释器只读 `/home/zhangyuheng/projects/ScratchVL/.venv/bin/python`，`PYTHONPATH` 指向本项目 `case-retrieval/round2/runtime`，`PYTHONDONTWRITEBYTECODE=1`。
- 首个待运行对照：`llm-relations-v2-fold0/prompts.jsonl`（16篇、zero-shot全文关系），随后`llm-extract-v2-smoke`（2篇11打包窗口）与每8概念证据批次；没有声称已产生LLM成绩。
- SapBERT FP16只读归一化模型从首日产物复制到同机自有 `models/SapBERT/`，传输进程的本地入口/源码位于外部setup目录；最终核验hash后才使用。
