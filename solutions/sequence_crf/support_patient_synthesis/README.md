# 已接受支持规则 + 患者六差集：唯一 CPU 合成

root明确授权，在两个独立结果已知之后，以已接受3049d9fe（Score .7542169692819372）为完整底，仅追加患者765cd5a0相对a179的精确6条关系差集。旧391保留断言只针对原患者397端点；新版支持gate已经删除2条旧public，不能把整份a179并入新版。

固定保留1061实体、36患者、410既有关系、54public支持、所有非关联原始字节、旧phenotype顺序和元数据；六差集与当前410/54的交集由实际输入记录，不预设为零。引用已审literal-span拼接，不解析模型回答、不训练或推断模型。两个CPU阶段各30秒＋TERM1；先assemble完整16pred与freeze，再evaluate唯一v2；旧metric直接复用，Score严格升才candidate，macro变动完整披露。首失败停止，不自动重试或扩臂。

源control.py b9ea4f50、plan edee73ab均在任何新pred或质量前冻结并提交推送。输入20份精确引用包含两独立终审，gold只复制已有SHA，在freeze之前不读正文/不重新hash。输出仓库外 `/workspace/patientphex-runs/round2-20261023/research/support-plus-patient-six-delta`，不存在即表示未开始，存在先查看终态，勿重复命令。当前未执行。

```sh
 timeout --signal=TERM --kill-after=1s 30s python solutions/sequence_crf/support_patient_synthesis/control.py assemble --plan solutions/sequence_crf/support_patient_synthesis/plan.json
 timeout --signal=TERM --kill-after=1s 30s python solutions/sequence_crf/support_patient_synthesis/control.py evaluate --plan solutions/sequence_crf/support_patient_synthesis/plan.json
```

这是反复使用的原dev开发合成，不是盲测。未达.8不合并实现，不官方提交。
