# 第二轮证据索引

保存小型指标、配置、源码/产物哈希、复现命令和少量错误分析。全量预测、模型和原始日志保存在台账指向的仓库外目录；此目录不存凭据或完整中间日志。当前尚无新运行证据。

- `qwen-download-complete.json`：13个固定版本文件最终size/SHA校验；`llm-smoke-manifest.json`为首个短/长输入GPU可用性证据。
- `llm-full-relations-fold0-*`：全文zero-shot关系负结果（同实体Score .615159→.569358）；包括解析回退、覆盖统计、运行及格式证据，不作为可部署改进。
