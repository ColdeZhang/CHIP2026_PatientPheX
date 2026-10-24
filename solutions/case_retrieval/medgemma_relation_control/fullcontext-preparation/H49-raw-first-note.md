# H49 新事实的窄范围检查

2026-10-24，第二轮训练结束、最终80推断前，只读检查已冻结 `formal-sources/infer_medgemma80_fullcontext.py` SHA `36d8185fe85f540f64411d0538e343fe564aeb36f03889a6ac624a0587ccfb66`。

L165–186 中生成后仅 native decode，完整 response 文本和 token 计数立即逐行 JSONL write、flush、fsync，然后才增加 completed。运行器没有候选 JSON/语义解析；唯一解析在本地 `evaluate_medgemma.py`，等全部80原始响应与manifest收齐并冻结后才执行。因此下游预期格式/语义错误不会丢失这条原 decoded response，不需要修改当前冻结源、重启或添加GPU诊断。

范围限定：当前证据保留 decoded 文本，不含全部 generated token IDs；native decode 或写盘本身失败时，当前行可能尚未写出。没有据本次检查宣称全部故障可无损恢复，也没有改变既定推断预算或重试策略。
