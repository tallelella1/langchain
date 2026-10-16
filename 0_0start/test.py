from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

DASHSCOPE_API_KEY = os.getenv("BAILIAN_API_KEY")
BASE_API_URL = os.getenv("BAILIAN_API_URL")

LLM = ChatOpenAI(
    model_name = "qwen3.8-max",
    api_key = DASHSCOPE_API_KEY,
    base_url = BASE_API_URL,
)

response1 = LLM.invoke("1+1=?")

print(response1)


