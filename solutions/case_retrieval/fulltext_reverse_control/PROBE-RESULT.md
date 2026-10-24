# 完整全文候选逆序：输入与最长探针

这是旧4f48适配器上此前未运行的完整全文逆序。已有短提示逆序、正序全文和四源并集保持各自历史结论，不混同新控制。

CPU唯一11.908秒完成：原80条/16篇/39患者/587概念/1557对保持，78条候选块反转、2条单候选自然不变；每条全文suffix原UTF8字节、患者集合、双逆全部字段精确。原生Qwen3_5Tokenizer实际全80 IDs/往返通过，normal670122→reverse670117 tokens；max14390、index19、cap14848，无截断，0GPU/0金标。

新最长probe原proc07b832aa于08:20:13启动并一次成功，program40.286秒/外层41.511秒。14390输入生成65token，9.698秒自然EOS，原strict全部CID/PID合法，0cap；加载6.739秒，峰值22,233,742,848B/预留23,502,782,464B，GPU1释放。原始IDs先持久，再保存Auto/native两种decode后判一致，再用原codec解析；无修复或重新生成。25原件包349f4a39/53,308B已全SHA恢复，summaryea5334b9/terminal06b535fd。

原模型/4f48、BF16/nativeSDPA、TF32 matmul=false/cudnn=true、greedy1600及EOS集合不变。正序基线来自A800，新逆序来自A30，硬件数值差异显式保留。探针不产生质量结论。

root据实测已授权一次完整80，1200+TERM30，精确复用本新reverse index19并生成79条；不使用旧normal缓存。原normal实际推断顺序是长度升序＋row_id升序，按源码9bb2b322第89行保持；准备期消息中的desc口述已纠正，未执行错误版本。新primary=全文normal∪全文reverse、diagnostic=reverse-alone，两完整预测共同冻结后各一次v2、旧0；对.676546354662594按Score严格升判断，macro完整报告。无自动F0/dev/官方后继。

[原始终态与恢复指针](probe/result-pointer.json)；全量日志/原始输出留仓库外。
