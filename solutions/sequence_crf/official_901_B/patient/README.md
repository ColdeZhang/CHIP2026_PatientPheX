# 当前901方案的官方B患者关系推断

目标是把已接受的第二份Qwen3.5 full64患者LoRA `f787f18d` 应用于实际B纯输入。不得使用原dev的六条新增关系作为B规则。保留原native BF16/SDPA、greedy batch1、14000输入/1600新token、EOS及有限缺键弃答政策；每行先持久原ID和raw，再分类。完整原epoch2三实际SHA与权重身份仍由runner校验，不训练或重新选择checkpoint。

`infer.py` 相对原 `memory_recovery/formal/final_infer.py` 只把固定dev输入SHA/64行/16文献换成实际B冻结合同，并修改产物标签。`source-delta.json`记录原/新SHA。`patient_first.py`、`limited_codec.py`、`raw_store.py`均为已审原文件的相同字节。

执行链只有一次：

1. 接收retrieval按原算法生成的normal-B完整纯输入prompt及full-LR oldpool（含原parser回退关系），分别绑定实际SHA。`prepare.py`做原patient-first转置和逆变换一致检查，保留code_map、患者及证据顺序；只加载原tokenizer，逐条对发布原生JSON的实际IDs与解码，无权重/GPU。超过14000即停，不能截短或删样本。
2. native census完成后冻结实际行数/文献/输入tokens、具体时间预算和源计划，提交推送，再由`driver.py`在已交接且fresh核空的A100 GPU2启动唯一完整推断。原任务目录、raw和已审dev均不修改；超时/失败保留原件，不自动重跑。
3. `export_positive_keys.py`逐件SHA核完整收件清单、完整行覆盖、生成/raw/decision三方一致及原始token独立解码，再重放同一有限codec。只导出显式模型 `[pmc_id, patient_id, original_concept]` 正关系与完整原code_map空间，不加入任何baseline或回退。
4. 与retrieval使用实际B和原canonical/alt-ID政策证明patient空间来自oldpool、public-new与其逐文献互斥。证明后才应用901的等价并集；有交集则按root决定恢复原counterfactual差集，不能临时删patient或猜。

当前native CPU420全输入已完成，原A100唯一推断proc_7c9baef1于07:17:15Z启动。具体SHA/绝对预算和只读恢复入口见路线WORKLOG及仓库外 `research/official-901-B/patient/`；未完整收件前不称结果已完成。root独占官方上传和最终统一审，禁止读取官方成绩。
