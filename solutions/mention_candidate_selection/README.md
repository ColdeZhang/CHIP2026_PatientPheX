# 固定 mention 公开候选选择

固定F1配方负结案；不扩F0或原dev，不改阈值。

强B 0.728697536469，全部接受实体并集 0.721025583801，新头替换 0.713183678719。两完整预测共同冻结后仅2次公共v2，原dev/官网0。

公开Sap Top8明确候选，Qwen3-8B监督生成互斥selected+missing/NIL/NO；全部2931训练和1416验证跨度保持固定。新增供给与整体精度、来源差异和恢复入口见[WORKLOG.md](WORKLOG.md)，完整结果指针见[evidence/terminal.json](evidence/terminal.json)。
