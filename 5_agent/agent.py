import os,requests,math
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage,ToolMessage,HumanMessage


load_dotenv(override =True)

BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_API_URL = os.getenv("BAILIAN_API_URL")

model = init_chat_model(
    model = "openai:qwen3.8-max",
    api_key = BAILIAN_API_KEY,
    base_url = BAILIAN_API_URL,
)

@tool
def get_weather(city: str) -> str:
    """获取指定城市最新发布的实况天气信息。

    Args:
        city: 城市名称，如"北京"、"上海"、"深圳"等。

    Returns:
        包含天气状况、当前气温和数据更新时间的信息。
    """
    api_key = os.getenv("SENIVERSE_API_KEY")
    if not api_key:
        return "请先设置环境变量 SENIVERSE_API_KEY，配置心知天气API私钥。"

    try:
        response = requests.get(
            "https://api.seniverse.com/v3/weather/now.json",
            params={
                "key": api_key,
                "location": city,
                "language": "zh-Hans",
                "unit": "c",
            },
        )

        data = response.json()

        # 心知天气的接口错误，例如密钥错误、城市不存在、权限不足
        if "status_code" in data:
            return (
                f"天气查询失败：{data['status']}，"
                f"错误码：{data['status_code']}"
            )

        response.raise_for_status()

    except ValueError:
        return "天气接口返回的数据不是有效的 JSON。"
    except requests.RequestException:
        return "天气请求失败，请检查网络连接和天气服务状态。"

    result = data["results"][0]
    weather = result["now"]

    return (
        f"{result['location']['name']}："
        f"{weather['text']}，"
        f"当前气温 {weather['temperature']}℃，"
        f"数据更新时间：{result['last_update']}。"
        f"数据来源：心知天气"
    )

@tool
def math(expression: str) -> str:
    """执行数学计算
    支持基本运算符（+、-、*、/、**）和常用数学函数
    Args:
        expression: 数学表达式，可以包含：
        - 基本运算：2 + 3, 10 * 5, 100 / 4
        - 幂运算：2 ** 10
        - 函数：sqrt(16), abs(-5), pow(2, 3)
    Returns:
        计算结果或错误信息
    Examples:
        calculator("2 + 3 * 4") 返回 "14"
        calculator("sqrt(16)") 返回 "4.0"
    """
    try:
        form ={
        "sqrt": math.sqrt,
        "pow": pow,
        "abs": abs,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "pi": math.pi,
        "e": math.e
        }

        result = eval(expression, {"__builtins__": {}}, form)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{str(e)}\n提示：请检查表达式格式，支持的函数有 sqrt,abs, pow, sin, cos, tan, log"

@tool
def get_time(city:str) -> str:
    """获取时间
    获取不同时区的时间,获取地名,返回时间

    args:
        zone:某个地区    
    """
    city_timezones = {
        "北京": "Asia/Shanghai",
        "上海": "Asia/Shanghai",
        "深圳": "Asia/Shanghai",
        "东京": "Asia/Tokyo",
        "纽约": "America/New_York",
        "伦敦": "Europe/London",
    }

    timezone_name = city_timezones.get(city)
    if not timezone_name:
        return (
            f"暂不支持查询{city}的时间。"
            f"当前支持：{'、'.join(city_timezones)}"
        )

    current_time = datetime.now(ZoneInfo(timezone_name))
    time_text = current_time.strftime("%Y-%m-%d %H:%M:%S")
    utc_offset = current_time.strftime("%z")

    return (
        f"{city}当前时间：{time_text}"
        f"({timezone_name},UTC{utc_offset})"
    )

#创建对话类
class SmatAssistant:
    #初始化
    def __init__ (self):
        self.model =model
        self.tools =[
                get_weather,
                math
            ]

        system_prompt = """你是一个多功能智能助手，可以帮助用户：
                        🌤 查询天气：使用 get_weather 工具
                        🔢 数学计算：使用 calculator 工具
                        """

        self.agent =create_agent(
            model =self.model,
            tools =self.tools,
            system_prompt=system_prompt
        )
        #历史聊天记录
        self.messages =[]

    def chat(self,user_input:str) -> str:
        #用户输入,传入历史对话后进行处理(包括AIMessage)
        self.messages.append({"role":"user","content":user_input})
        result =self.agent.invoke({"messages":self.messages})

        #提取历史记录的AIMessage,倒叙遍历,提取出最新的AIMessage
        self.messages =result["messages"]
        for msg in reversed(self.messages):
            if msg.type =="ai" and msg.content:
                return msg.content
            
        return "抱歉，我无法处理这个请求。"

    def reset(self):
        """重置对话历史"""
        self.messages = []

def main():
    assistant =SmatAssistant()

    print("=" * 40)
    print("🤖 多功能智能助手（LangChain 1.2）")
    print("=" * 40)
    print("\n我可以帮你：")
    print(" 🌤 查询天气")
    print(" 🔢 数学计算")
    print(" ⏰ 时间查询")
    print("\n输入 'quit' 退出，输入 'reset' 重置对话\n")

    while True:
        user_input =input("👨用户:")

        if user_input.lower() == 'quit':
            print("再见！👋")
            break

        if user_input.lower() == 'reset':
            assistant.reset()
            print("✅ 对话已重置")
            continue

        if not user_input.strip():
            continue
            
        response = assistant.chat(user_input)
        print(f"🤖 助手: {response}")



if __name__ == "__main__":
    main()