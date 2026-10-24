# 独立响应代码 cyclic1 控制

只对已存在的 PT 输出代码层做一个固定循环移位，不改临床患者 ID。原患者索引 `i` 获得 `PT((i+1)%n+1)`；如原 `PT001→p0, PT002→p1, PT003→p2`，新表为 `PT001→p2, PT002→p0, PT003→p1`。患者原顺序、CID/HPO 原顺序、system、全部临床 user 前缀和 marker 保持，只有 PT 对应关系及 user 末尾映射表改变。

本地结构结果：80 行、16 文献、39 患者、587 CID、1557 对；**53 行需新 NN，27 行单患者保持原 JSONL 整行字节，可复用原 IDs/raw/canonical**。17 个结构合同通过。tokenizer、模型及质量尚未运行。状态见 [台账](WORKLOG.md)，完整源及外部产物见 [清单](evidence/source-manifest.json)。

```bash
PYTHONDONTWRITEBYTECODE=1 python solutions/response_code_equivariance/prepare.py \
  --coded /workspace/patientphex-runs/round2-20261023/research/patient-response-code-readiness-20261024/held.response-code.jsonl \
  --clinical /workspace/patientphex-runs/round2-20261023/research/qwen35-sft32-full-report-feasibility/expanded80.jsonl \
  --output /path/to/new-empty-preparation-directory
```

实际准备已完成，不要在原目录重复执行。外部 `research/response-code-equivariance-v1/native-preparation-v1/` 含四个 CPU 源文件、两份冻结 gzip、逐行结构合同和原 held native 审计；`native-plan.template.json` 精确绑定全部文件及 .100 **历史**环境预期。主线程部署到新私有目录后，可将该模板复制为 `native-plan.json`，由指定 Python 执行 `native_driver.py --plan <plan> --output <新终态目录>`。driver 固定 CPU 120 秒＋TERM 5 秒，隐藏 CUDA；runner 在实际运行时核 runtime、六项 tokenizer 资产及 native/Auto/template/双 decode，结果前不宣称 token 数或本次环境已测。当前没有执行该入口。

后续复用 C32 adapter `2c777e88…`，不训练。严格 JSON 策略沿原 parser，只换明确的 cyclic1 逆映射；无 set-decoder 修复。两预锁端点及 B4 对照见 [策略](evidence/endpoint-policy.json)。不同 host/runtime 的结果只称开发控制，不作纯映射因果推断。
