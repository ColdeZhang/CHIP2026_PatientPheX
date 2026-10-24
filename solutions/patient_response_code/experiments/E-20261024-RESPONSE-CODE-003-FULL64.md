# E-20261024-RESPONSE-CODE-003-FULL64

目标：为内部双折均正的固定四来源关系组合提供 fresh full64 coded 组件；不把coded-alone称全流程最高分。本路线不打开原开发集gold、不执行quality；组合/唯一质量由root预锁流程负责。

固定原 short normal265（原train64目标27ecf476），保全部clinical文本/markers、CID/HPO及次序，仅沿原PT表示转换；原dev使用既有full-report64输入4621fec5。训练265/64docs/173患者/1908CID/5328对，held64/16docs/36患者/466CID/1006对；全部候选祖先排原dev16。不是负结案expanded全文训练，不从C32adapter续训。

实际CPU .100 proc53d9fb9c在13:25:41.491 rc0/28.590秒，native27.836秒：train918058tokens、17399监督（原15410）、max6462/idx137/125监督，另一Paretoidx207为4800/128；held523821tokens/max13425idx1。完整329行原生IDs、Auto模板、decode、监督mask通过；真实Torch2.7.1与预定A1002.6差异披露，A100模型运行前重核实际IDs/资产。原A100仅部署fetch和read recovery LUNA，CPU从未start；.100是独立attempt2。

已授权唯一A100240+TERM30 H51：原epoch1首8真实Adam→3累积含idx207→最长idx137，共12FB/52201tokens，状态丢弃。若实际finite/三族更新/base冻结/显存余量及成本<=5400，自动fresh2ep/68calls/尾1每轮/warmup ceil(.05×68)=4；共享5400+30含两完整SHM→.100 state ACK。全部标准CE/r16/lr1e-4/seed20261023不变。之后完整64fresh1200+30、input13824/gen1600、strict/raw-first只交pure组件，无新评分。

真实GPU尚未启动。源/计划与恢复位置见 full64/CPU-readiness.json 与外部 patient-response-code-full64-readiness-v1；后继actual有据追加，原CPU/失败收据保持不可变。
