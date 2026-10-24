# 接手注意

V2增加第三视图时，首次宽泛替换`return result`把list断言也插入了动态module loader，导致CPU在创建tokenizer前TypeError、无输出目录/模型/质量。原737dd193 source/plan保留；恢复源仅移除loader错位两断言，真正views三视图门保持。不能仅以py_compile通过当运行接口有效；修改同名通用返回点需按函数精确定位。失败证据external v2-first-interface-failure.json，恢复用plan-v2-recovery.json，不再执行原v2入口。

Warm PEFT路径必须从copy收据逐成员相对路径解析：checkpoint目录不等adapter目录。本次首probe误缺/adapter，加载前5.225秒失败/0forward/0update，保原失败；独立path-recovery只补已存子目录，仍原240绝对截止15:15:27.964586。
