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

class WeatherSchema(BaseModel):
    city : str = Field(default="北京", description="城市名称")
    if_forecast : bool = Field(default=False, description="是否包含明日天气预报")

#定义工具
@tool("get_weather_if_forecast", description="查询当日天气，可以包含明日天气预报", args_schema=WeatherSchema)
def get_weather(city:str,if_forecast:bool):
    res = f"{city}今天天气不错"
    if if_forecast is True:
        res= res+",明天天气也不赖"
    return res

print(convert_to_openai_tool(get_weather))

model_with_tools = model.bind_tools([get_weather])

messages = [HumanMessage("今天天气如何？明天呢？")]
response = model_with_tools.invoke(messages)
messages.append(response)
tool_calls = response.tool_calls

for tool_call in tool_calls:
    if tool_call["name"] == "get_weather_if_forecast":
        tool_msg = get_weather.invoke(tool_call)
        messages.append(tool_msg)
final_response = model_with_tools.invoke(messages)
messages.append(final_response)
for msg in messages:
    msg.pretty_print()