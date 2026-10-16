from langchain_core.messages import SystemMessage,HumanMessage,AIMessage,ToolMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
# 从.env文件中加载环境变量
load_dotenv(override=True)
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_API_URL = os.getenv("BAILIAN_API_URL")
model = init_chat_model(
model="qwen3.8-max",
model_provider="openai",
api_key=BAILIAN_API_KEY,
base_url=BAILIAN_API_URL
)

def recent_messages(messages, max_pairs=3):
    """
    保留所有系统消息和最近 N 轮对话。
    适用于 HumanMessage、AIMessage 交替出现的完整对话。
    """
    if max_pairs < 0:
        raise ValueError("max_pairs 不能小于 0")

    # 分离系统消息和对话消息
    system_msgs = [
        m for m in messages if isinstance(m, SystemMessage)
    ]
    conversation_msgs = [
        m for m in messages if not isinstance(m, SystemMessage)
    ]

    # 每轮两条消息；0 表示不保留对话
    recent_msgs = (
        conversation_msgs[-(max_pairs * 2):]
        if max_pairs > 0 else []
    )

    return system_msgs + recent_msgs

long_conversation = [
    SystemMessage("你是一个善于给出通俗易懂解释的数学老师")
]

long_conversation.append(HumanMessage("解释一下正弦定理"))
r1 = model.invoke(long_conversation)
long_conversation.append(AIMessage(r1.content))

long_conversation.append(HumanMessage("解释一下余弦定理"))
r2 = model.invoke(long_conversation)
long_conversation.append(AIMessage(r2.content))

long_conversation.append(HumanMessage("解释一下勾股定理"))  
r3 = model.invoke(long_conversation)
long_conversation.append(AIMessage(r3.content))

question = HumanMessage("我问第一个问题是什么")

print(f"完整历史消息数：{len(long_conversation)}")
print(f"问题：{question.content}")
# 用新列表追加当前问题，避免修改原始历史、影响后面的裁剪对比。
full_response = model.invoke(long_conversation + [question])
print(f"完整历史回答：{full_response.content}")

messages_to_send = recent_messages(long_conversation, max_pairs=2)

print(f"\n裁剪后历史消息数：{len(messages_to_send)}")
print(f"问题：{question.content}")
recent_response = model.invoke(messages_to_send + [question])
print(f"裁剪后回答：{recent_response.content}")
