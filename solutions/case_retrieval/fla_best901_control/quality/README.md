# 完整四流FLA部署终点

只评测一个端点：原901流水线的normal、reverse、public-new、patient全部234条换为本次fresh FLA响应。对照为已接受901的缓存分数`.7543613981990362`，旧评测零次；新完整预测只运行一次公共v2 all/text。Score严格提高才作为待独审候选，macro完整披露，不设额外macro门；未提高则关闭本配置，不扫描子集或组合。

`template.json`默认禁用，实际GPU终态未绑定。必须四流全部完成、零缓存，并核回收目录中`terminal.files`的每一成员SHA和原逐行IDs/raw记录后才可绑定。binder只解析JSONL外层证据，不解释`response`内容。父线程完成实际绑定和源码检查点后运行：

```bash
python bind_actual.py --template template.json --inference-dir /absolute/collected-root \
  --published-plan /absolute/published-gpu-plan.json --source evaluate.py --output /absolute/bound-plan.json
python run_quality_once.py --plan /absolute/bound-plan.json --source evaluate.py \
  --output-dir /absolute/unique-quality-owner
```

回收根保留GPU的`plan.json`、五份源、`terminal.json`及其全部相对成员，含每流`actual/<flow>/responses.jsonl`、record-manifest、IDs、raw、summary、complete、backend、assets、generation及四份native输入audit。绑定完整actual后，该目录也保存独占评测标记。唯一CPU预算60秒、wall65秒、TERM5秒；失败保留证据，不重试。

`adapt_assembly.py`从已成功0.109740秒复装901的原`1b414126`源码提取`assemble` AST。只把原native fingerprint函数替换为已核actual回调，把两条旧patient decisions及status-count身份断言替换为新观察回调。逆向还原这三处后，整个函数AST必须与原函数完全一致。其余函数直接来自原模块及其已冻纯函数来源，原源文件不改。源码准备会静态生成这份AST证据，不执行四流解析或组装。

固定关系算法仍为normal排序→旧HGB6401支持public追加→reverse字面保真union→LGB1061支持替换→追加`patient−重建的旧门控a179`。没有硬编码原35/54/6条正关系；历史产物SHA和这些正例数量不再限制新响应。原normal/reverse CID级fallback、public失败CID弃答、patient `60731`部分PID弃答及非法整行弃答均保持；完整达到1600 token上限的输出仍交原策略处理，不因来源校验而强加自然EOS或全行strictJSON门。截止时中断的四流不能进入质量终点。

仅写出一个最终完整预测，保持16文献、36患者、1061实体和全部非association原字节。预测、组装记录、patient codec观察全部fsync，并保存完整freeze后，才注册H50 validator、加载评测器并首次读取gold；metrics先写入再检查分项不变量。没有原生与FLA关系的附加组合，也不评分任何中间预测。

对照仍是多次使用过的原dev开发集。原normal/A30、reverse/public/A800、patient/A100与本次全部A30的硬件和运行时存在混杂，结果是完整部署变体控制，不能单独归因为FLA内核。
