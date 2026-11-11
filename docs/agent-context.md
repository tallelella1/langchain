# LangChain 学习工程接续信息

更新：2026-10-15。本轮只整理历史与工程入口，没有重新安装依赖或调用模型。

- 当前项目 `D:/grow/langchain`，旧的 `C:/Users/tallerella/Desktop/langchain` 已不存在。
- 已核对目录：`0_0start`、`1_langsmith`、`2_message_prompt`，根目录有 requirements.txt 和 .vscode/settings.json。另一个教程目录 `C:/Users/tallerella/Desktop/langchain1.2_tutorial` 是不同目录，不混写。
- `D:/conda/envs/langchain/python.exe` 当前存在。10 月 13 日对话记录曾将该环境调整到 Python 3.13.12，并报告依赖核对完成；本轮未复核版本、包完整性和 Notebook 内核。
- .vscode/settings.json 当前只指定 Conda 环境管理器，没有为每个 Notebook 固定内核。先看当前内核，历史中曾选到 MSSQL，不能用终端安装成功推断内核正确。
- 最近学习任务是消息对象、deque、保留最近对话及流式输出。用户要求完整代码并保持与原例相似，不额外添加判断标志或功能；该要求适用于教学复刻，不代表工程代码一概省略必要校验。
- `2_message_prompt/2-1-1.py`、`2-1-2.py` 及多个 Notebook 当前存在；历史提到过的 `1.py` 当前不在该目录，接续时先确定目标。
- 服务商和模型按当前任务确认；历史出现过 Moonshot／百炼／OpenRouter，不能互相套用凭据和地址。密钥仅在本地环境中使用。

## GitHub 仓库

- 2026-10-16 创建私有仓库：https://github.com/tallelella1/langchain。
- 本地远端 `origin` 指向该仓库，使用 `main` 分支。
- `.env` 仅保留在本地；`.gitignore` 排除凭据、Python 缓存及 Notebook 检查点。

## DataClean 异步数据处理平台（2026-10-27）

- 新项目位于 `D:/grow/langchain/dataclean`，入口与完整运行说明见 `dataclean/README.md`。本任务用户选择不启用 superpower 流程。
- 已实现注册/登录、JWT 与 Argon2id、销售 CSV 校验清洗、金额统计、权限隔离、任务状态、分页历史、失败重试和四种结果下载，附网页工作台。
- 两种运行方式：Windows 本地 SQLite + 数据库队列线程；Docker Compose 使用 PostgreSQL + Redis + Linux Celery + 独立 dispatcher，自动执行 Alembic 迁移。
- 幂等键由数据库唯一约束保证；任务使用原子抢占、可续租租约与执行令牌防止旧结果覆盖；输出目录全部生成后再发布。
- 验证：SQLite 和独立 PostgreSQL 测试库各 26 项通过；两种模式真实 HTTP 全流程通过；Redis 中断恢复与 API 重启后的记录/报表持久化通过；浏览器注册、上传、自动刷新与下载通过。
- PostgreSQL 测试有一条 Starlette 关于未来迁移 httpx2 的弃用提示，不影响本次通过结果。测试库独立，不操作业务库。
- 当前 Docker 栈保留运行，工作台 `http://127.0.0.1:8000/`。停止命令在 dataclean 目录执行 `docker compose down`，保留数据卷。
- 项目 `.venv` 已创建，新增依赖装在其内（现有虚拟环境复用父 Conda 环境的已装包）；根 requirements.txt 未修改。`.env` 包含随机本地密钥，已忽略，不记录其值。
- 完整说明包含 CSV 字段、异常/去重规则、API、测试与面试要点。面向单机演示，尚未实现用户配额、自动文件清理和完整公网运维措施，不调用模型 API。

