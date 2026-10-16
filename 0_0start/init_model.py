from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import os

load_dotenv(override=True)

BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
BAILIAN_API_URL = os.getenv("BAILIAN_API_URL")

model = init_chat_model(
    model = "openai:qwen3.8-max",
#porvider(:前的参数，也可以单独写出model_provider) 看之前用openai、openrouter还是deepseek这种官方方法，具体有什么方法可以看init_chat_model的源码
    api_key = BAILIAN_API_KEY,
    base_url = BAILIAN_API_URL,
)

response1 = model.invoke("简单介绍自己")

print(response1)


