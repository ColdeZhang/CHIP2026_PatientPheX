# 患者混合目标：仅恢复框架调用接口

原probe-v1首row62/2704tokens已完成模型到完整BF16 head前向，随后 `row_loss_scope.py:22` 的复合参数guard失败；core未调用、无反向/Adam/质量，rc1/72.035秒，原11件和GPU释放收据完整保存。不能把CPU数值合同或静态审查当真实框架接线已通过。

实际安装源码：PEFT `peft_model.py` d22a1c50，forward2085–2171默认传递 `output_attentions=None`、`output_hidden_states=None`、`return_dict=None`；TF `generic.py`222da931的`can_return_tuple`先pop return_dict；实际Qwen b4b17fe0在1726把其余kwargs透传loss。原guard无差别拒绝kwargs，不是损失数值失败。原失败没有记录各参数局部值，不能把源码推导冒充原GPU实测局部变量；恢复新增实际call观察，在任何参数断言之前持久保存。

`row_loss_scope_v2.py`仅允许两个已证诊断名字且值必须None，未知键/任一非None仍拒绝，num_items/ignore_index/shift_labels原guard保持。`mixed_loss.py` dc98bb97、真实12输入4cfdb1f1、模型/精度/完整head、P87/S16/g8和目标α.5均不变。旧源不覆盖，新remote namespace为 `sequence-round2/patient-balanced-alpha05/probe-interface-recovery-v2`。

root已明确独立fresh240+TERM30恢复预算；不从原72秒失败state续训，不延长或重写旧绝对deadline。先新的纯Python调用接口合同（不是Torch/数值重评），源/计划push，再fresh空卡门启动唯一完整12反向工程probe。三族B变化、完整Adam/梯度、最长5556/释放仍需真实终点；正式2ep/32calls/warmup2、质量均未启动。

部署仅传变更源码/计划/接口收据，其他已验输入/公共helper在原自有同服务器目录逐SHA复制到新目录并fsync/reopen；不重编码labelbank/训练数据或搬模型权重。失败、UNKNOWN与LUNA均按实际收据保留，不自动重启。
