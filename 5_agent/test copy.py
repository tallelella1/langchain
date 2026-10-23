from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
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

agent = create_agent(model = model,
                    system_prompt = SystemMessage(content = "你是一个小学生，对数学一窍不通")
)

resp = agent.invoke({
    "messages": [
        {"role": "user", "content": "5的阶层等于多少？"}
    ]
})
rprint(resp)