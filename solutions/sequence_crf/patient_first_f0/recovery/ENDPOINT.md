# 独立 23+35 abstention recovery 的 CPU 终点

这里只准备源码和冻结计划，未运行 assemble/evaluate，未打开新 58 条 raw、gold/targets、模型权重，也未评分。原 strict 作业失败且未评分的记录、原 endpoint-plan/policy/source 保持不变。

`endpoint.py` 与 `policy.json`、`endpoint-plan.json` 已绑定实际新 recovery protocol、infer-plan 和 proc_d19bf1ac 的 infer-job 原字节。policy 先保存，source 硬绑定 policy SHA，plan 再绑定 source；未来 common-freeze 绑定全部证据。plan 中 policy 的部署位置是父目录 B/policy.json，准备件本身仍在本目录，owner 复制原字节即可。

部署时把 endpoint.py 放在 exact limited_codec.py 和 patient_first.py 同目录；把本目录 policy.json、endpoint-plan.json 原字节复制到 B。endpoint 可保留在当前目录，也可复制到 Git 自有 recovery 目录；source 校验只要求内容 SHA 精确。B 是 `/workspace/patientphex-runs/round2-20261023/research/patient-first-closed-set-F0/formal-preparation/abstain-recovery`。不要改写已绑定的 infer-job 原收据。

待 owner 完整回收后，使用未加 `-O` 的 Python 分别运行：

```text
python3 /path/to/endpoint.py assemble --directory B
python3 /path/to/endpoint.py evaluate --directory B
```

这两个命令此处均未执行。assemble 的总 CPU cap 为 30 秒，evaluate 两臂总 cap 为 60 秒；任一错误留下证据并停止，不覆盖已存在 assembled/evaluation，不评分 partial。

assemble 要求成功完整 collection、wrapper rc0/实际卡释放和 58 行标记，162 个原件全部 SHA 精确。raw-index/decision/raw/generation 的远端 ref 仅映射到已回收 terminal 的相对路径；不会访问远端。原 23 行必须保持完整 8853B 原字节前缀，其后恰为 35 行；58 个 row ID/实际排序/输入 token 数完全绑定原 58 输入。

每条原 token ID 用固定 tokenizer.json 独立完整 decode，仅存在实际末 EOS 时去掉该末 token，无 EOS 保留全部文本。独立调用同一 limited codec，结果必须逐字典等同保存的 decision；cached 23 仍须 complete。partial 只贡献显式合法 PID/CID 正关联，遗漏键和整行拒绝都保留 abstention 计数，没有补写模型空列表。预测格式中的空 phenotype 仅表示没有选中的正关联。

primary 保留 old9B 的全部 947 实体、非关联字段和 47 位患者元数据，使用 415 候选/1166 对内的显式正关联。secondary 原字节复用已审 build_union 函数，等于 oldSFT48 与 primary 的字面并集。两份完整 fullpred、codec-replay、assembly-report 及全部源/原计划/新 protocol/job/terminal 引用共同冻结。

evaluate 只能在共同冻结存在且每项 SHA 仍精确时执行。先建立排他的 before-gold 标记、核两份格式，再首次打开 gold。沿公共 v2 的 all/text 接口按 primary、secondary 固定顺序各一次；old9B 指标仅复用。keep 固定为 delta(micF1+macF1)>0 且 delta(macF1)>=0，未通过即关闭，不扫阈值/组合、不迁 dev。

`static-receipt.json` 记录 0.042635 秒的纯 AST/stdlib 源检查：relations/build_union 与原源 byteexact，实际六个部署源 SHA、protocol/plan/job 及 25 个原引用绑定、共冻→before-gold→固定两评的静态顺序均核过。没有重新跑 codec 27 例/旧 tests/tokenizer 或任何未来 endpoint。未来数据内容和质量仍待 owner 实际完整终点。

`source-snapshots/` 和 snapshot-manifest.json 保留本次读取的小源/计划原件。全预测、raw、tokenizer/HPO、日志和权重应留仓库外；Git 只需小源、policy/plan、README、静态收据及最后 owner 摘要。
