# Full64 原开发集纯推断源码

源码准备完成，尚未执行绑定、远端操作、分词或模型。仅在 fresh short265/full64 两轮正式训练产生完整两份持久 ACK（epoch1/global34、epoch2/global68）及真实 campaign-complete 后绑定 epoch002 adapter。`bind_after_training.py` 不创建训练回执，也不接受旧C32训练状态；trainer 固定为相邻 `../formal_train/train.py`，fingerprint schema 固定 `patientphex-qwen35-full64-response-code-indexedHead-lora-v1`。

输入是已冻结 `held.response-code.jsonl`（SHA `b76e4b3de00432c692c0cefd84baa44275cf62404e9887f1e051ec0982db1f9b`）：64行、16篇原dev文献、36患者、466 CID、1006候选患者概念对；原dev输入SHA `c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e`。不读取开发集答案，不生成评测入口或最终实体预测；完整 canonical 记录供owner按自己的冻结候选与融合约束装配。

原生CPU证据位于 readiness 的 `native100-attempt2/native-terminal/`：result SHA `0ef29c1252c53527e8ee6947d05006e7179edd60a62dcfaea3f530ac87b2d627`，audit SHA `51e7a3327d09966cc08a250571e03d3a97705a5e53a0abe2a062e810d4704adb`。CPU使用 Torch2.7.1+cu126；推断目标A100使用 Torch2.6.0+cu124、Transformers5.20.2、PEFT0.21.2、tokenizers0.23.1。运行时必须在权重加载前重新编码64行，先保存实际输入hash核对表，再逐行确认与实际CPU一致。总523821 tokens，最长原index1为13425，拒绝超过13824的输入；不截断、不删提示。

单次全64 fresh greedy，1600新token上限、1200秒+TERM30。禁止复用旧响应。标准模型KV缓存保持旧实现。生成IDs先fsync保存再decode，原响应先fsync保存再strict parse；原strict PT→clinicalPID decoder三个函数AST完全复用。无自然EOS、达到生成上限或任意键/类型/患者码错误均整行弃答，不修复、不补负答案。

以下CLI在训练终态真实成立后由owner执行。controller目录可替换为实际完成的那个目录；inference目录必须全新且位于仓库外。当前本地默认示例路径需与owner最终训练controller位置一致。

```bash
python solutions/patient_response_code/full64/inference/bind_after_training.py \
  --controller /workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-formal-v1/controller \
  --directory /workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-formal-v1/originaldev64-inference-v1 \
  --readiness /workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-readiness-v1 \
  --dev-input /workspace/patientphex-data/day1/dev16.input.jsonl
python solutions/patient_response_code/full64/inference/operations.py deploy --directory /workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-formal-v1/originaldev64-inference-v1
python solutions/patient_response_code/full64/inference/operations.py start --directory /workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-formal-v1/originaldev64-inference-v1
python solutions/patient_response_code/full64/inference/operations.py inspect --directory /workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-formal-v1/originaldev64-inference-v1
python solutions/patient_response_code/full64/inference/operations.py collect --directory /workspace/patientphex-runs/round2-20261023/research/patient-response-code-full64-formal-v1/originaldev64-inference-v1
```

远端独立推断目录固定 `patient-response-code/full64-v1/originaldev64-inference-v1`。唯一start intent建立后，未知状态先inspect原作业，不重复start。训练adapter仍从A100原 `/dev/shm/patientphex-response-code-20261024/full64-formal2ep/epoch-002/adapter` 读取，并逐文件核ACK哈希；若该原副本已清理，应由owner明确恢复同一字节身份后再运行。

成功终态为 `COMPLETE_RESPONSE_CODE_FULL64_TWO_EPOCHS_FULLTEXT64_FRESH_CANONICAL`。收集后 `terminal/canonical.jsonl` 按原64行顺序完整输出，每行含 `row_id`、`pmc_id`、`candidate_patients`（CID→原clinical PID列表）、`strict_complete`、`abstention`、`provenance=fresh`、`original_index`；弃答行保留记录且 `candidate_patients={}`。伴随保存全部raw、IDs、实际EOS/cap状态和input-ID对照，不将“完整覆盖”解释为“所有行解析成功”。

当前验证仅为7个Python源静态编译、原strict decoder AST一致、既有CPU JSON和无答案held输入的stdlib census核验。没有声称完成绑定或真实推断。`source-manifest.json` 记录新增源与旧源身份；父任务统一负责台账和Git。
