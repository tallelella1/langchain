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
