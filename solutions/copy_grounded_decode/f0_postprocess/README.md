# Copy F0 完整后处理（仅CPU准备完成）

本目录由 document_relations 临时准备，copy_decode_readiness 统一接管、commit 和实际执行。未读取 F0 gold，未解析实际F0响应、运行模型或评分。当前只有两个有区分力合成合同；未来真实47收件仍必须通过完整验证。

固定 `B4_F0 ∪ (validnewG ∩ (oldG ∪ S ∪ Q ∪ C))`，16doc/47patient/947entity、B343关系及所有非关系字节保留。S/Q含旧B时，装配先证明其同一继承B⊆B4且⊆S,Q，再证明去旧B前后新增集合一致；不将旧B额外当成新来源。无support计数扫描，固定literal v2 atoms、精确NIL与NO排除。

相对F1 fallback_v2，F0已预锁通用整行弃答并由新runner正常rc0完成，故检查 `inference_reported_success`、`complete47_actual_members_verified`，EOS/cap/strict数量由真实终态验证，不硬编码F1的rc1/37strict。任何cap/nonEOS/原strict失败整行零新增，严格成功才走原Grounded解释器；没有前缀恢复或逐项挽救。

`D=R/research/copy-grounded-decode-v1/F0-v1`，`C`为方案根目录；`G`为原grounded_generation目录。依次：

1. 原 `C/f0/collect.py` 完整收件后，`python postprocess.py extract-bind`；以G worktree为PYTHONPATH执行本目录 `extract_quotes.py --plan extraction-plan.json`。
2. 原 `G/f0/held_prepare.py --extraction-manifest D/quotes-v1/manifest.json --output-dir D/normalization-preparation`。
3. 必要query非零才执行本目录 `normalization100/bind.py --preparation-manifest ... --release-evidence <本47真实release> --fallback-decision C/f0/decision.json`，计划push后由owner沿remote pack/deploy/唯一start/observe/collect；实际namespace `copy-grounded-decode-v1/F0-v1/Sap100-v1`，FP16 GPU180+30，原41099bank/b59不变。这里未执行任何remote动作。
4. `postprocess.py normalize-bind`；G worktree PYTHONPATH执行原 `G/f0/held_finalize.py --plan normalization-finalize-plan.json`。
5. `postprocess.py assemble` 完整冻结唯一pred；freeze推送后 `postprocess.py score` 至多一次v2，byteexact继承0。无alone/有效子集评分。

`support-protocol.json`绑定真实现有B4/G/S/Q/C及继承B；`f0/decision.json`由主owner的prepare生成，本目录不抢写。所有真实运行输出均仓外，旧源不改。
