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
        """初始化知识库检索器、LLM 请求地址和认证请求头。

        输入：无。
        输出：无；实例状态通过属性保存。
        """
        self.retriever = WikiRetriever()
        self.url = f"{SILICONFLOW_BASE_URL}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json",
        }

    def _call_llm(self, messages: list) -> str:
        """调用 SiliconFlow 对话模型并提取文本回答。

        输入：`messages`，符合 Chat Completions 格式的消息列表。
        输出：模型返回的回答字符串。
        异常：HTTP 请求失败或响应缺少预期字段时抛出异常。
        """
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.1,
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

