# 天池登录会话的使用与恢复

本页记录恢复方法；实际登录状态、核验时间和比赛权限只维护在 [WORKLOG.md](../../WORKLOG.md)。当前环境使用 Chromium 与 Python Playwright。按用户指定使用 `ALI_ACCOUNT` 与 `ALI_PASSWD` 环境变量中的用户名、密码；`ALI_PHONE` 仅在需要手机验证或用户明确选择手机号登录时使用，所有值均不输出。

## 保存位置

- 浏览器 profile：`/workspace/.local-state/tianchi-browser`，目录权限 `700`。
- 恢复快照：`/workspace/.local-state/tianchi-storage-state.json`，文件权限 `600`。
- 当前浏览器 CDP：`http://127.0.0.1:9222`，只监听本机。

这些都是仓库外的认证资料，不能提交到 Git、复制进实验产物或打印其内容。环境重建或服务器会话到期仍可能使它们失效；Git 中的文档备份不会备份登录凭据。

## 恢复顺序

1. 优先用 `playwright.chromium.connect_over_cdp("http://127.0.0.1:9222")` 连接已有浏览器，在原 context 中操作。结束一次工具调用时保留该浏览器及 context。
2. 浏览器未运行时，先确认没有进程占用 profile，再启动同一 Chromium profile。使用 `--user-data-dir=/workspace/.local-state/tianchi-browser`、`--remote-debugging-address=127.0.0.1`、`--remote-debugging-port=9222`。本环境需要 headless 模式。已观察到跨轮次后浏览器退出、只剩僵尸进程的情况；不能假定端口持续可用。仅在确认没有活跃进程占用该 profile 后，才处理残留的 `SingletonLock/SingletonSocket/SingletonCookie` 链接。
3. 如需恢复到新 context，使用 `browser.new_context(storage_state="/workspace/.local-state/tianchi-storage-state.json")`，打开天池官网并核验。快照已实际通过独立 context 的登录恢复检查。
4. 登录成功后，以 `umask 077` 和受限目录保存 `context.storage_state(path=..., indexed_db=True)`，确保文件权限为 `600`；不要输出该方法返回的数据。若使用新 context 工作，后续刷新其快照。
5. 会话过期后使用官方登录流程，优先登录用户指定的用户名账号；短信验证码由用户提供，仅用于当次认证。手机号入口可能选择另一个绑定账号，不能用它替代指定的用户名账号。不能通过轮询、无意义提交或修改客户端过期时间承诺永久登录。

## 切换账号

在当前 context 中打开阿里云官方退出地址 `https://account.aliyun.com/logout/logout.htm?oauth_callback=<编码后的天池回跳地址>`，回到天池核验 `getSession` 已为未登录；退出确认后移除旧恢复快照，防止误恢复旧账号。再从 `https://tianchi.aliyun.com/home/auth/aliyun/login?site=cn` 进入账密登录，使用环境变量。新账号通过天池登录、报名权限及快照恢复验证后，保存新的快照。仅删除天池单域 Cookie 不等于退出阿里云 SSO。

## 每次提交前的只读核验

在已打开的 `https://tianchi.aliyun.com` 页面内用 `fetch`，携带当前会话并设置超时：

| 接口 | 只读取/记录的字段 |
|---|---|
| `/home/user/getSession` | `code == "SUCCESS"` 且 `data.login == true` |
| `/v3/proxy/competition/api/race/getDetail?raceId=532512` | `signed`、`showSubmit`、`showStsSubmit`、`currentSeasonId` |

官方前端展示提交入口的条件为 `signed && (showSubmit || showStsSubmit)`。登录有效不代表已报名或有当前赛季的提交权限；需检查实际页面。提交页路径为 `/competition/entrance/532512/submission`，只查看页面不消耗评分次数，选择文件和实际提交则需遵守额度及实验记录要求。

只输出白名单状态，不打印完整 session/detail 响应、Cookie、请求头或异常堆栈。当前环境曾发生 Playwright `APIRequestContext` 直连 IPv6 不可达；浏览器页面内的请求可用，见 [HANDOFF.MD](../../HANDOFF.MD)。

真正提交前，还需核对比赛、赛季、输入数据、文件 SHA-256、格式和剩余次数；提交后保存提交标识及官方结果。没有新预测文件时不上传归档产物试跑。

## 首日提交工具与凭据

[提交工具](../../scripts/tianchi_submit.py) 只连接已有登录浏览器，逐份核验并提交冻结 JSONL；入口 JSON 保存文件、SHA、字节数、赛季、分支/代码版本。必须指定独立 receipt 路径，已有凭据会阻止再次上传。选择文件前先落盘尝试记录，选择一次后等历史记录及额度减少，超时标记待人工检查，不能自动重试。

当前页面历史表仅有“文件名称 / 提交时间 / 状态”；工具先核验列结构，只记录这三项，不读取分数页。首日平台接收清单在 [day1-20261023.json](../submissions/day1-20261023.json)。下一会话按用户通知回收分数时，用清单中的平台文件名和提交时间匹配，避免把旧提交的成绩混入新路线。

## 回收官方成绩

用户要求回收成绩时，访问 `/competition/entrance/532512/score`。已观察到页面只读请求 `GET /v3/proxy/competition/api/race/score/my?raceId=532512&pageNum=1&pageSize=10`；核对 `currentSeason.seasonId`，按 `raceScoreVO` 分页读取。只保留 `id`、`fileId`、比赛/赛季、提交时间及 `jsonValObject` 中13项 P/R/F1/Score；不能记录账号信息或认证字段。UI四位小数不可代替API完整精度。

在提交页加载时观察 `/v3/proxy/competition/api/race/upload` 的响应，`raceFile` 提供文件 id、名称、时间和大小。该页面初始化使用 POST 读取元数据；通过加载页面观察即可，不手工重放上传操作。用 `score.fileId == raceFile.id` 连接，再与保存的完整平台文件名、时间、字节数和本地 SHA 核对。首日五份的[匹配证据](../submissions/day1-20261023-official.json)已保存。

CDP连接恢复后的视口可能变回窄屏，从而没有桌面“我的成绩”菜单；需要在同一次连接中设置视口再操作，或直接导航已确认的score路由。只读刷新不消耗提交额度。
