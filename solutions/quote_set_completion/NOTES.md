# 接手注意

V2增加第三视图时，首次宽泛替换`return result`把list断言也插入了动态module loader，导致CPU在创建tokenizer前TypeError、无输出目录/模型/质量。原737dd193 source/plan保留；恢复源仅移除loader错位两断言，真正views三视图门保持。不能仅以py_compile通过当运行接口有效；修改同名通用返回点需按函数精确定位。失败证据external v2-first-interface-failure.json，恢复用plan-v2-recovery.json，不再执行原v2入口。
