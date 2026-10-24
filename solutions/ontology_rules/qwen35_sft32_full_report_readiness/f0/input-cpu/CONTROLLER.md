# 唯一 F0 native CPU 控制器（准备完成，未执行）

`run_cpu_once.py` 只在负责人显式选择模式时导入 MCP client。这里未导入 MCP、未访问远端、未运行 native CPU、未读 gold 或执行模型/质量。

固定 mount `69b991ee-183f-4add-af1a-579af92479c9`、UID 1045，自有远端目录 `ontology-round2/qwen35-sft48-full-report-f0-cpu-v1`，private `ontology-round2/qwen-venv/bin/python`，`a800-migration/runtime`。CUDA 隐藏，输入源 `358a2359…` 和 CPU plan `124bc87c…` 保持不变。

负责人备份源码后唯一运行：

```bash
python run_cpu_once.py --mode run
```

7 个原字节文件（plan 的 6 项 + plan）已打包为 `source-inputs.tar.gz`，499119 字节、8 个最大 65536 字节分片。实际执行先独占本地 deployment intent，再独占创建远端私有目录和 remote intent；随后上传分片，远端核全部分片/包/7 成员 SHA、逐源码 compile 和 fsync。只有完整部署之后，先独占本地 CPU intent、再独占远端 CPU intent 才启动 CPU 子进程。内层 `timeout 60s`、TERM 后 5 秒强制终止，外层 MCP `run_command` 75 秒。日志/child PID/完成状态保留在原目录。

任何 UNKNOWN 或中断都保留原 intent，重复 `--mode run` 会拒绝；没有自动重试启动。恢复只使用：

```bash
python run_cpu_once.py --mode inspect
python run_cpu_once.py --mode collect
```

inspect 只读原 namespace 的状态和文件身份；collect 只接收原 rc0 完整结果，不进入 deployment 或 CPU launch。若 CPU 尚未完成或失败，collect 拒绝，不重新运行。对于本地尚无 CPU intent 的异常，先 inspect 并由负责人核对原状态，不能自动补发启动。

回收 `result.json`、`row-audit.json`、`CPU-complete.json` 的原字节：远端固定三成员 zip，按最大 49152 字节分片读取，核包 SHA/长度、每片 SHA/长度、每个文件 SHA/长度以及 result 中的 row-audit 身份。直接落本地 `input-cpu/` 根；`actual-result.json` 是 result 的同字节别名。已回收文件只有同字节才可复用，不覆盖不同证据。`probe_input_gate_passed=false` 是如实回收的输入上限结果，不能由控制器绕过。

`deployment-manifest.json` 记录固定 7 件和 8 分片；`controller-static-checks.json` 记录本地 controller/六个远端片段 compile、包内逐成员 SHA 检查，无 MCP 调用。`controller-preparation.json` 是本阶段清单。压缩包、未来原工具回执/日志/结果留仓库外；仓库只保存本说明、源码、清单和检查收据。不要把本地准备或工具返回成功误写为实际 CPU 通过。
