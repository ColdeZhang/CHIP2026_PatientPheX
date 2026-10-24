# 新全文训练模型的固定 F1 验证

已完成两轮完整持久化、全部80条fresh推断及两端点唯一评测。primary .675134214／diagnostic .672875653均低于强基线 .676546355，按事前Score门负结案；见 [结果](RESULT.md) 与 [实际引用](result-pointer.json)。

保留原binder/template/runner用于复现来源，不能重新执行已完成的训练、推断或质量入口。原14400输入上限／1600 greedy、raw-first／严格整行解析、旧idx19缓存禁用、两pred共冻等合同均保持。没有自动F0／原dev／官方迁移授权。
