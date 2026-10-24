# 单条全文逆序工程探针

只运行当前 CPU 实际最长的一条，不预设 index 或长度。`bind_cpu.py` 读取已经核验的 `result.json`、`row-audit.json`、`longest-probe.jsonl`，按实际长度重新选最长及 512 向上 cap，再生成可执行 plan；完整 80 提示引用沿 CPU 结果，单条逻辑 SHA 必须匹配审计。父线程负责实际绑定、提交及唯一启动。

```text
python bind_cpu.py --template template.json --cpu-dir <本地已回收CPU小目录>
  --remote-cpu-dir <服务器CPU actual目录> --output <新probe部署目录/plan.json>
<固定Python> launch.py --plan <服务器唯一probe目录/plan.json>
```

旧全文正序 probe `062cbf90` 和 full80 `9bb2b322` 的真实调用为 Qwen3_5 text-only/BF16/native SDPA、原 4f48 LoRA、greedy/use_cache、1600、EOS 配置∪tokenizer。实际 flags 是 matmul TF32=false、cudnn TF32=true。新入口显式保持这些值；未沿用 R1 的配置或 parser，也不因官方 B 同名模型推断等价。`legacy_runner.py` 全字节沿原33a5；`unique_object/parse_complete` 沿原全文源码，恰好全部 CID、合法唯一字符串 PID；缺键或非法输出整行弃答，无临时修复或 fallback。

新增工程证据包括：CPU 选中身份绑定、发布 native 双编码/双解码核、原始 IDs→完整 raw→原 strict 解析、生成中每32 tokens partial、父链 fsync、绝对 240＋TERM30（含资产校验和加载），210 秒软停止保留证据。实际EOS不变；正常未到时间门时仅增加观察。失败目录保留，不重启或扩预算。

部署携带本目录6个 Python 和固定 model-source-manifest，实际 base15文件及 adapter两文件重新全 SHA。A30 native模块及生成helper SHA 沿已核 runtime，禁止 FLA 替换。原正序 A800 与新逆序 A30 的数值硬件差异显式保留，不能称严格相同硬件对照。

当前只完成6份源码 compile、原 codec函数和legacy source字节核。没有执行 binder、tokenizer、远端、GPU或评分。模板仍 disabled；完整80和质量不由本入口自动启动。
