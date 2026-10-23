# 天池登录会话的使用与恢复

本页记录恢复方法；实际登录状态、核验时间和比赛权限只维护在 [WORKLOG.md](../../WORKLOG.md)。当前环境使用 Chromium 与 Python Playwright；手机号从 `ALI_PHONE` 环境变量读取，不输出其值。

## 保存位置

- 浏览器 profile：`/workspace/.local-state/tianchi-browser`，目录权限 `700`。
- 恢复快照：`/workspace/.local-state/tianchi-storage-state.json`，文件权限 `600`。
- 当前浏览器 CDP：`http://127.0.0.1:9222`，只监听本机。

这些都是仓库外的认证资料，不能提交到 Git、复制进实验产物或打印其内容。环境重建或服务器会话到期仍可能使它们失效；Git 中的文档备份不会备份登录凭据。

## 恢复顺序

1. 优先用 `playwright.chromium.connect_over_cdp("http://127.0.0.1:9222")` 连接已有浏览器，在原 context 中操作。结束一次工具调用时保留该浏览器及 context。
2. 浏览器未运行时，先确认没有进程占用 profile，再启动同一 Chromium profile。使用 `--user-data-dir=/workspace/.local-state/tianchi-browser`、`--remote-debugging-address=127.0.0.1`、`--remote-debugging-port=9222`。本环境需要 headless 模式。
3. 如需恢复到新 context，使用 `browser.new_context(storage_state="/workspace/.local-state/tianchi-storage-state.json")`，打开天池官网并核验。快照已实际通过独立 context 的登录恢复检查。
4. 登录成功后，以 `umask 077` 和受限目录保存 `context.storage_state(path=..., indexed_db=True)`，确保文件权限为 `600`；不要输出该方法返回的数据。若使用新 context 工作，后续刷新其快照。
5. 会话过期后使用官方登录流程。可从环境变量取得手机号；短信验证码由用户提供，仅用于当次认证。不能通过轮询、无意义提交或修改客户端过期时间承诺永久登录。

## 每次提交前的只读核验

在已打开的 `https://tianchi.aliyun.com` 页面内用 `fetch`，携带当前会话并设置超时：

| 接口 | 只读取/记录的字段 |
|---|---|
| `/home/user/getSession` | `code == "SUCCESS"` 且 `data.login == true` |
| `/v3/proxy/competition/api/race/getDetail?raceId=532512` | `signed`、`showSubmit`、`showStsSubmit`、`currentSeasonId` |

官方前端展示提交入口的条件为 `signed && (showSubmit || showStsSubmit)`。登录有效不代表已报名或有当前赛季的提交权限；需检查实际页面。提交页路径为 `/competition/entrance/532512/submission`，只查看页面不消耗评分次数，选择文件和实际提交则需遵守额度及实验记录要求。

只输出白名单状态，不打印完整 session/detail 响应、Cookie、请求头或异常堆栈。当前环境曾发生 Playwright `APIRequestContext` 直连 IPv6 不可达；浏览器页面内的请求可用，见 [HANDOFF.MD](../../HANDOFF.MD)。

真正提交前，还需核对比赛、赛季、输入数据、文件 SHA-256、格式和剩余次数；提交后保存提交标识及官方结果。没有新预测文件时不上传归档产物试跑。
