# Seq2seq患者集合：静态准备入口

结论、约束、选型与未完成事项见[REPORT](REPORT.md)。唯一选定base，不下载/并跑large。全部原件在：

`/workspace/patientphex-runs/round2-20261023/research/seq2seq-patient-set-readiness-20261024/`

- `public/`：原论文/小配置/两平台元数据、SP与fast tokenizer；large仅safetensors JSON头，无tensor payload。
- `census-v1/`：87训练+39纯输入完整CPU成本与逐行哈希。
- `prepared-v1/`：`train.jsonl.gz`、`held_input.jsonl.gz`及manifest；复制到新host后文件内容SHA必须一致。
- 源码：`census.py`与`prepare.py`已本地CPU执行；`probe.py`仅编译，未跑任何模型。

后续由root先取得一次probe所需授权/资源，按[probe-plan](probe-plan.json)从ModelScope固定revision取得**一套base**资产并核四件SHA（当前没有下载权重），确认目标环境Torch与Transformers4.50.0；不能将本文件当已启动。部署同一供给manifest与train文件，路径可迁移、内容不可变。大权重仍在仓外。

实际有界入口模板（需要把变量绑定为当时真实路径和空闲UUID）：

```bash
timeout --signal=TERM --kill-after=30s 600s "$PYTHON" probe.py \
  --plan probe-plan.json --supply "$SUPPLY/manifest.json" \
  --model-dir "$MODEL" --output "$NEW_OUTPUT" --gpu-uuid "$FREE_UUID" \
  --admit-until "$LAUNCH_EXPIRES_UNIX" \
  --authorized-probe-only four-shapes-two-adam-no-formal
```

`LAUNCH_EXPIRES_UNIX`必须在实际发布前生成、最多300秒；过期入口在load前拒绝。未知启动先恢复同一请求，不另开。输出排他创建，每例先保存数值诊断再断言；真实OOM/超时保留，不改shape或延预算补称成功。工程4FB/2Adam与短训练例解码均不评质量，完整87训练/39held另行冻结计划。

本任务无正式训练命令、无模型输出或分数。`probe-plan.json`预算是预案；真实8k/16k耗时/峰值尚无，不能从架构矩阵数推出实耗。
