# 纯 Python 接口恢复合同

源码与计划在 `c785f2467d4ee706d983c005989495d6260c0a2a` push/远端exact确认后已唯一执行：**27case全PASS，10接受/17预期拒绝，23次新hook observer先落盘，scope全恢复，rc0、0.216996秒、未超时**。原预算10秒＋TERM1，独立目录 `R/research/patient-balanced-mixed-loss-alpha05/interface-recovery/python-contract-v1`；result `d76b7da8…`、terminal `043ba6cd…`、完整观察 `33a87317…`。原 GPU 失败、旧 hook及317项数值合同保留，没有重跑。`interface_contract.py` 的显式输入为 `--plan`、`--output`，输出位置须与冻结计划相符；既有终态禁止重新执行。

绑定真实提取 PEFT forward `021d3f71…`、return decorator `731c6ba6…`、来源manifest `78e66621…`、Qwen完整源码 `b4b17fe0…`、旧hook `97a0c5ce…` 与父线程新增hook `7b9900fc…`。AST只抽真实函数，保持函数body，去掉Qwen的 `auto_docstring` 元数据装饰器；原 `can_return_tuple` 确实执行。Tensor、text backbone、LM head、outputs及loss/core均为纯Python sentinel/recorder，不import Torch/Transformers/PEFT，不读真实目标，不调用任何数值loss。

27个固定case包含：g4/g8下原PEFT默认两diagnostic `None` 经真实Qwen透传、新hook接受而旧 `not kwargs` 拒绝；显式两个None和return_dict=None；return_dict=False原decorator取走并转tuple；直接无kwargs；未知key（含None）、任一diagnostic非None（True/False/0）、num_items/ignore_index/shift_labels及dtype/device/vocab/sequence guard拒绝。每次新hook先调用observer，observer立即fsync实际字典及“core尚未调用”，随后case保存全部观察才作断言。通过路径核logits/labels/native callable的对象identity、完整metadata内容、g/P87/S16/alpha.5不变，以及退出后原callable恢复、scope清除。

范围限普通LoRA、非prompt/POLY/MPT、无aLoRA及特殊adapter参数：`get_alora_offsets_for_forward` 明确使用identity stub，adapter forward-hook上下文为nullcontext；不宣称检查其内部实现。该合同只证明接口传参及guard生命周期，不能证明真实tensor数值、checkpoint backward、LoRA/Adam更新、GPU显存或质量。已过的317项数值合同沿用，不重算。

所有实际observer、case观察、源码抽取位置、result、child/intent/stdout/stderr/terminal均本地保存并fsync。实际没有failure文件；失败路径保留代码没有本次运行失败。该终态仅证明上述纯Python接口合同，不代替后继独立GPU恢复终点。
