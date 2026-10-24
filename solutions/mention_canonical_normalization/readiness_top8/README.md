# 公开Top8候选供给准备（独立后继，不改2353负结论）

只用冻结全部2353 train32目标/1284 exactquote，公共41099 Sap bank Top8唯一合法HPO；不插gold/exact、不用train48别名，不读heldgold，不做LLM训练或质量。先核训练原子cover@1/@8、compound部分/全覆盖、NO/NIL、OTHER及竞争组。>32token整行保留而不截短编码，empty候选显式记原因。

历史Biomed上下文Top8 CE已有正式负；其公开+train48 alias/gold追加、独立候选binary与本拟公开纯Top8多选NO/NIL/OTHER并不相同。9B entity-slot已训评的是给定span-ID二分类、固定.5/重叠decoder，非Top8组内选择。本准备只证可竞争供给，不能由coverage推断新LLM质量。

沿旧public-bank.py/semantic.py原字节复用6asset与公共bank；.100仅独立1284 query一次180+30，不重编码bank或下载，不干预2931。参数109482240；FP16 backbone+FP32 CLS/L2是原真实接口，不称整个encoder FP32。原freeze2353 train_ref与完整资产见plan.json。外存R/research/mention-public-top8-readiness-20261024。

实际CPU长度1284全核max21，全部≤32，不需要改变原native32长度或丢弃任何行。首120预算在GPU启动前由root改180+30，未重probe。

全1284 query已23.374秒完成，全部2353行训练census见REPORT.md。27件411347B/9b1b63ff小包完整远端核验并回下载同SHA，精确locator见backup.json。
