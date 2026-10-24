# 全文SFT32实际CPU结果

两项均完成，未加载模型、使用GPU、读取验证gold、训练或评测。真实训练尚未授权。

- native 124行/32文献：22.406947秒；原402594/max5523 → 全文1009181/max13549（唯一index98），token比2.506696573，平方长度和比6.047085928。512向上工程cap13824，0截断，model context262144。
- 发布原生/Auto/template/双decode逐行核；原encoded7fa9、全文0755d5fc；原与全文7170个assistant监督IDs逐行不变。
- 同目标Torch2.7.1+cu126/TF5.20.2 selected-CE合同2.329901秒，288基本例+2完整head梯度+4梯度累积+4忽略位置扰动+4custom ignore通过。logit梯度最大差0；loss最大绝对差7.62939453125e-6，在声明相对/绝对容差内，不能称所有loss逐字节相等。

旧局部两轮1149.460秒乘实际token比得到2881.348秒，仅线性成本代理。长序列attention、完整BF16 head、backward/checkpoint与optimizer状态的真实GPU成本尚未验证，平方长度增长更大，不能把4500+30建议或6166旧探针说成全文训练已可行。完整case日志/row audit留仓库外，Git仅摘要与精确指针。
