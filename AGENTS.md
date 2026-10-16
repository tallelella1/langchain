# LangChain 学习工程协作规则

- 根目录为 `D:/grow/langchain`。按需读 `docs/agent-context.md`；先确认用户指定文件及是否有未保存的编辑内容。旧桌面路径只作为定位线索。
- 这是学习工程。按示例复刻时优先保留命名、结构、交互和讲解顺序，只替换本次目标需要的部分；不默认增加菜单、标志位、框架或额外功能。需要时使用 `teaching-code-match`。
- `0_0start` 包含模型初始化与调用示例，`1_langsmith` 包含相关学习内容，`2_message_prompt` 包含消息与提示词示例。不要按旧回答中已不存在的文件名创建替代文件，先找当前文件。
- 目标 Conda 环境为 `langchain`，候选解释器 `D:/conda/envs/langchain/python.exe` 当前存在。运行或安装前确认其实际版本和内核，不能把系统 Python 或 Notebook 的 SQL 内核当作它。
- 字典消息与 `SystemMessage`／`HumanMessage`／`AIMessage` 对象使用各自接口；检查消息顺序、最近轮次的定义及当前问题是否进入请求。
- 模型接口先确定实际服务商、base URL、model ID 和环境变量名；直连 Moonshot、百炼和 OpenRouter 分别核对。环境变量值留在本地，不写入报告或日志。
- 环境问题按需使用 `python-env-triage`。依赖清单以当前 requirements.txt 为准；不自动升级／降级 Python 或改写固定版本。
- 仅安装或环境核验任务，以版本、导入和依赖一致性为验证范围；源码行为改动使用受影响的离线检查。真实模型请求可能收费，只有属于本次授权与验收范围时才调用。
- 交付区分语法／离线检查与真实模型响应；未调用 API 不宣称真实对话通过。
