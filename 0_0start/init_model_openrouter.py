from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL")

model = init_chat_model(
    model = "openrouter:apodex/apodex-1.1-mini:free",
    api_key = OPENROUTER_API_KEY,
    base_url = OPENROUTER_API_URL,
)

response1 = model.invoke("简单介绍自己")

print(response1)


