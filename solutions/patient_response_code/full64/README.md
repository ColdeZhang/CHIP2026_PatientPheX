# Response-code full64 准备

当前仅获 CPU 120+TERM5 授权。沿原短 normal265 训练目标做原 PT 转换；原开发集使用既有完整全文64纯输入。不得从C32继续训练，不得把expanded全文训练替代原short训练。全部原clinical文本、患者属性、CID/HPO、次序保持；只有明确system说明、尾部映射表和训练answer患者值改变。

数据/真实执行在外部 `research/patient-response-code-full64-readiness-v1`，入口 `prepare.py`；`cpu/native_cpu.py` 使用原编码助手与原codec，逐完整329行核原生tokenizer/Auto/template和监督mask，输出真实长度、监督tokens及二维Pareto。候选祖先和训练64均排原开发集16；不读取原开发集标签、不评分。

CPU源/计划先提交，再 `cpu/deploy_via_bridge.py` 原stage→serve→ready→fetch，`cpu/start_collect.py start` 唯一启动，终态 `inspect-collect`。未知mutation先查原namespace，不盲重复。后继GPU新shape/训练/推断预算尚未授权。

更新：实际329 CPU已完成，见CPU-readiness.json。Root现授权一次240+30新Pareto H51，通过且估时<=5400可接fresh2ep两ACK，再完整64纯组件推断1200+30；本owner仍无原dev quality授权。该更新不修改此前CPU-only冻结plan。
