from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import os

# 沿用之前笔记的百炼配置；初始化本身不会发送模型请求
load_dotenv(override=True)

BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_API_URL = os.getenv("BAILIAN_API_URL")
model = init_chat_model(
    model="qwen3.8-max",
    model_provider="openai",
    api_key=BAILIAN_API_KEY,
    base_url=BAILIAN_API_URL,
)

@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

@tool(parse_docstring=True)
def get_news() -> str:
    """
    获取当日新闻
    """
    return "近期，受全球储蓄芯片短缺等多重因素影响，多地回收商称废旧手机回收市场迎来“火热潮”，回收价格普遍上涨，旧手机成“香饽饽”。"


model_with_tools = model.bind_tools([get_weather, get_news])

messages = [
    HumanMessage("今天杭州天气如何？今天新闻是什么？别瞎编")
]

response = model_with_tools.invoke(messages)
response.pretty_print()

messages.append(response)

for tool_call in response.tool_calls:
    if tool_call["name"] == "get_weather":
        tool_msg = get_weather.invoke(tool_call)
    elif tool_call["name"] == "get_news":
        tool_msg = get_news.invoke(tool_call)
        print(tool_msg)
        messages.append(tool_msg)
    else:
        raise Exception("不存在的工具")

final_response = model.invoke(messages)

messages.append(final_response)

for msg in messages:
    msg.pretty_print()



