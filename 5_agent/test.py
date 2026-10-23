from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv(override=True)

from rich import print as rprint

# 以init_chat_model为例
model = init_chat_model(
    model = "openai:qwen3.8-max",
    api_key = os.getenv("BAILIAN_API_KEY") ,
    base_url = os.getenv("BAILIAN_API_URL"),
)
agent = create_agent(model)

resp = agent.invoke( {
    "messages": [
        {"role": "system", "content": "你是一个小学数学老师，耐心，幽默，讲解深入浅出"},
        {"role": "user", "content": "5的阶层等于多少？"}
    ]
})
rprint(resp)