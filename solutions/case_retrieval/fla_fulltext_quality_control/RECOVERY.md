# 首次入口失败与限定恢复

原唯一作业09:25:09启动，27.150秒外层 /24.624秒程序退出，0forward/0生成/0质量，GPU已释放。新backend额外要求每个模块标签为FLA的包装器源码都位于FLA wheel；此假设比已通过A合同更强。旧A实际图e8a6f955已存在FLA模块标签、源码在Torch `_dynamo/eval_frame.py` 的包装器，故该新增门会错误拒绝此前已验证运行。原失败未保存完整graph，不能假称已读取这次失败的具体wrapper。

CUDA隐藏的一次CPU导入显示native分支而非GPU下的FLA分支，因此仅保存为这个诊断方法的范围限制，不把空差集当GPU修复证明。恢复直接回到已审A的合同：wheel全部.py逐字节、两FLA闭包/原native unwrap、hub源码、native卷积均保留；只撤新增inspectfile归属假设，不扩大白名单。真实新graph将在恢复的actual依赖记录中保存。

独立recovery目录保留原infer逐字节，只有backend上述删项、launcher固定原09:45:09.929177 UTC/monotonic3725518.161365813截止及部署目标变化；1200秒时钟不重置，soft截止仍提前30秒，TERM后30秒。恢复仍全部80fresh，无原native回应缓存、模型/模板/解码变更。

收件器初次替换前缀有误，在读取不存在terminal时失败，尚未创建archive；原intent/错误收据留collection-wrong-prefix-preserved。仅修前缀后冻结原真实终态一次，不重模型或造新的原始响应。
