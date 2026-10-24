# 完整901与完整FLA的唯一缓存并集

原四流全替换在独审b952104c后按计划负结案（.753895452309<.754361398199），不修改该冻结结论。root随后明确授权此独立控制：对完整原901和完整新fba773f7取通用患者关系并集，只追加新集合中原901没有的全部关系；不挑选其对错，不逐流搜索，不硬编码已观察的2项差异。依据是此前F1全文的N∪FLA亦正、F0并集同正；此处是看过原dev替换结果后的新研发控制，不能叫盲测或事前独立验证。

复用39c3efe0已验证的通用`values/relations/union`纯函数及651f9dcc字面字段定位，只AST加载所需定义，不运行旧CLI。保原901所有非association字节、原患者和phenotype顺序；compound HPO拆原子供key去重、NIL/非HPO分号按原literal政策。原16篇/36患者/1061实体身份全核，源检查只跑成熟通用边界fixture，不提前组装真实预测。

唯一完整新预测与追加清单fsync后freeze，再H50注册validator、格式检查、首次gold读取、1次公共v2 all/text；旧两个metrics缓存只读。成功门仅Score严格高于原901 .7543613981990362，macro完整披露，否则关闭无扫描。CPU60秒、wall65+TERM5、零fit/NN/官方。`plan.json`固定输入与来源，运行前checkpoint push。

执行：`python control.py --plan /absolute/plan.json`。原始终态、完整预测和结果均在仓库外独立`research/fla-best901-whole-union-20261024/`。
