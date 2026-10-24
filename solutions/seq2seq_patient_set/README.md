# LongT5 患者集合工程准备

唯一选定 `google/long-t5-tglobal-base`，原强G的87条quote答案语义不变。正式训练未启动；单base资产与有界工程已完成，`.100 GPU0` 在2026-10-24 17:05:05.631 UTC确认释放。选型、许可和数据限制见[REPORT](REPORT.md)，实际原件refs见[probe-results](evidence/probe-results.json)。

仓外原件根：`/workspace/patientphex-runs/round2-20261023/research/seq2seq-patient-set-readiness-20261024/`。

- **当前唯一供给** `quote-prepared-v2/manifest.json`，SHA `641403ea7c923edfde2bbf3b2f0d28a22113e5062b84634c1c59320a4d4d8ec1`；完整87训练+39纯输入，逐行原答案/传输答案/labels双重以上SHA。`prepare_quotes.py`和`quote_codec.py`为当前准备入口。
- `prepared-v1/`、`census-v1/`及`probe-plan.direct-id-not-selected.json`仅是未选direct-ID静态对照，不进入当前模型。
- `public/`保留原论文、固定HF/MS小资产与元数据；large仅safetensors JSON头，无tensor payload。
- `collected/cpu-setup/`与`collected/assets-base/`保留实际单base全SHA和原生加载身份检查；`collected/probe-v1/`保留0FB加载guard失败，`collected/probe-v2/`保留全部恢复工程输出和释放证据。

远端根：`/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/seq2seq-patient-set-v1`。只有一套`assets-base/`，不需要把990MB权重回传root。实际Python `/home/zhangyuheng/projects/ScratchVL/.venv/bin/python` 为3.10.12，Torch2.7.1+cu126；只读隔离`runtime-t450/`为Transformers4.50.0/tokenizers0.21.4/hub0.29.3。精确PYTHONPATH在[evidence/asset-runtime.json](evidence/asset-runtime.json)。不能修改共享原Python环境。

本次冻结执行源 `9eec22c349be6662bad4428798ca2f7687300fec`。执行入口 `probe_driver.py` 调用外层timeout与 `probe.py`，四完整形状、前两例真实Adam，随后16token训练例解码，无held/质量。恢复使用独立`probe-v2`，`--term-deadline 1792861959.3071957`，`--admit-until 1792861791.2954605`；原期限现已失效，输出目录排他创建，不能将文档当成重启授权。

后继正式load必须复用 `probe.restore_shared_input_identity(model, output_dir)` 的原生共享恢复与前后值/身份guard：原checkpoint独立247587456参数，TF4.50/Torch2.7 assign加载会将输入共享权重呈现为三个Parameter。恢复不得改LM head或任何权重字节。原失败及CPU修复证据不能删除。

工程结果：4FB、2Adam、短训练解码均完成，295梯度tensor全部有限。8k×213 FB0.746s、峰allocated6.696GiB；16k×3 FB1.763s、峰allocated10.828GiB/reserved17.129GiB。两例不代表16k×658联合最坏形状，也不代表完整epoch/完整生成耗时。探针状态已丢弃，无checkpoint保存恢复或模型质量结论。
