# 原生四流复装901

目标是证明新的CPU组装入口仍实现已接受901的原方法。这里只从冻结候选、Task1实体、四路原生响应开始；不声称原文候选生成、权重推断或整个上游重训已经复现。

`plan.json`绑定64条normal、64条reverse、42条public-new、64条patient提示和响应，全部路径可迁移后在合同中重新绑定。`contract.json`记录每个真实文件的完整SHA；更完整谱系见`inventory-pointer.json`。准备阶段只读取源码、元数据和旧预测封装，尚未运行复装、gold、模型或评分。

固定流程如下：

1. normal与reverse沿原`817fdc`解析器，逐CID缺失、非法或截断时保留原full-lr回退；public-new沿同一解析器，但失败CID弃答。保留数值PID转字符串的旧行为，不新增全行严格JSON门槛。
2. normal关系按原规则排序，追加原HGB6401实体支持的public关系，随后沿`39c3`的复合HPO原子化、前缀保持规则追加reverse关系，重新得到旧门控a179关联数组。
3. 抛弃0e2c模板内的关联数组，将重建数组填入其非association字节，核0e2c完整SHA。只使用该模板已经冻结的1061个Task1实体及其他字段。
4. 从同一public原始正关系投影LGB实体支持，以原`replace_subset`替换旧支持子集，核3049完整SHA。保持其余关系的原值、列表顺序和元数据。
5. 原patient `60731` limited codec重新解释raw+实际IDs：遗漏PID只对遗漏部分弃答，非法整行弃答，无baseline回退。复核原decisions缓存；只追加`P−a179_oldGate`中尚未出现在新支持基线的关系，核901完整SHA。

原native预期：normal333、reverse305、public raw145、旧支持35、新支持54、a179391、patient318；患者差集6，最终416关系。35和6均从响应与支持集合重新计算，不是写死用于生成的key名单。三个预期SHA只用于结果核对，入口不读取3049或901预测作为生成输入。实体、患者、文献及全部非association原字节保持1061／36／16。

源码从已冻结文件AST中只装入明确列出的纯函数、类和常量；不导入旧评测CLI。原patient逐token解码证据复用已完成的`9fc90` endpoint及`2b362` codec replay，此次CPU入口不加载tokenizer。父线程另行完成234提示的native tokenizer检查。

父线程完成源检查点后，唯一原生复装入口为：

```bash
python assemble.py --plan plan.json --output-directory /absolute/unique/native-replay-v1
```

入口有60秒期限，新目录与独占写入防止重复；保留`intent.json`、`prediction.jsonl`、`observations.json`、各关系集合及`reproduction.json`。首次失败保留证据并报告，不自动重试；不调用gold、评测器、fit、模型或网站。

此plan只适用于原native缓存。后续FLA四流必须先绑定实际完整GPU证据，再沿同一纯组装函数执行；原native的历史SHA、关系数量和decisions相等断言不能移作新模型质量门槛。新raw保持旧字段接口：前三流需要`row_id/pmc_id/response/hit_token_limit`；patient额外保留`generated_token_ids/output_tokens/removed_demonstrations`。不得用FLA全篇实验的严格CID codec替换原patient limited codec，也不得把`P−a179_oldGate`改为无条件全并。
