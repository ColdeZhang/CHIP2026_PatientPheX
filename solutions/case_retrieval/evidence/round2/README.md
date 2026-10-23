# 第二轮证据索引

保存小型指标、配置、源码/产物哈希、复现命令和少量错误分析。全量预测、模型和原始日志保存在台账指向的仓库外目录；此目录不存凭据或完整中间日志。CPU内部选型、固定dev复评与Qwen负结果均保留对应证据。

- `qwen-download-complete.json`：13个固定版本文件最终size/SHA校验；`llm-smoke-manifest.json`为首个短/长输入GPU可用性证据。
- `llm-full-relations-fold0-*`：全文zero-shot关系负结果（同实体Score .615159→.569358）；包括解析回退、覆盖统计、运行及格式证据，不作为可部署改进。

- `lora-resume-independent-review.json`：真实Adam/scheduler/RNG状态恢复独立审查；连续数值轨迹等价未建立。
- `sft-extract-fold0-formal-manifest.json` / `sft-extract-fold0-training-complete.json` / `sft-extract-fold0-final-checkpoint.json`：train48-only305窗2epoch与最终adapter身份；原始状态/权重外置，推断再做实际流式hash。
- `sft-extraction-raw-paired-summary.json`：原90窗严格paired跨度收益和6条触顶，非Score。
- `qwen-linked-control-summary.json`：零样本实体HPO完整负控制，明确区分固定关系消融与重建k21/.35完整管线。
