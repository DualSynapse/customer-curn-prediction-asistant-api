from langchain_google_genai import ChatGoogleGenerativeAI
from src.config.settings import get_settings

settings = get_settings()
LLM_API_KEY = settings.GEMINI_API_KEY
LLM_MODEL = settings.GEMINI_AGENT_LLM_MODEL
LLM_TEMPERATURE = 0.1

model = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE
)