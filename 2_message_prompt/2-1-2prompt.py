from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 从 .env 文件中加载环境变量
load_dotenv(override=True)
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_API_URL = os.getenv("BAILIAN_API_URL")
model = init_chat_model(
    model="qwen3.8-max",
    model_provider="openai",
    api_key=BAILIAN_API_KEY,
    base_url=BAILIAN_API_URL,
)

EXIT_WORD = "exit"
MAX_PAIRS_HISTORY = 3

# 消息队列
messages = [
    SystemMessage("你是一个没有耐心，不友好的叫小米的电商客服")
]


def keep_recent_messages(messages, max_pairs=3):
    """保留系统消息、最近 N 轮完整对话和当前用户消息。"""
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    conversation_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
    # 当前用户消息尚未收到回复，所以需要额外保留这一条。
    recent_msgs = conversation_msgs[-(max_pairs * 2 + 1):]
    return system_msgs + recent_msgs


print(f"✨ 请输入问题，输入 {EXIT_WORD} 结束对话\n")

# 5. 多轮对话主循环
# 轮次记录
i = 1
while True:
    print("\n", "=" * 10, f"-> 第 {i} 轮对话开始 <-", "=" * 10, "\n")
    user_input = input("🙋 请输入：")

    # 退出判断
    if user_input.lower() == EXIT_WORD:
        print("🌙 对话已结束，欢迎下次再来！")
        break

    # 追加用户消息
    messages.append(HumanMessage(content=user_input))

    # 流式输出模型回复
    print("🫩  小米：", end="", flush=True)
    reply_content = ""

    # 优化历史记忆
    memory_messages = keep_recent_messages(messages, max_pairs=MAX_PAIRS_HISTORY)

    # 控制发送给模型的消息长度
    for chunk in model.stream(memory_messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            reply_content += chunk.content

    print("\n", "=" * 10, f"-> 第 {i} 轮对话结束 <-", "=" * 10, "\n")
    i += 1

    # 追加 AI 回复
    messages.append(AIMessage(content=reply_content))
