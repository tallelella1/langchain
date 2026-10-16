from langchain_openrouter import ChatOpenRouter
from dotenv import load_dotenv
import os

load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL")

LLM = ChatOpenRouter(
    model_name = "apodex/apodex-1.1-mini:free",
    api_key = OPENROUTER_API_KEY,
    base_url = OPENROUTER_API_URL,
)

response1 = LLM.invoke("1+1=?")

print(response1)


