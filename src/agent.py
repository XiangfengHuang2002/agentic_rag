import requests
from src.config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    LLM_MODEL,
)
from src.retriever import WikiRetriever


class GameAgent:
    """底层业务助手：负责检索、构造 prompt 和调用 LLM。"""

    def __init__(self):
        self.retriever = WikiRetriever()
        self.url = f"{SILICONFLOW_BASE_URL}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        }

    def _call_llm(self, messages: list) -> str:
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.1,
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

