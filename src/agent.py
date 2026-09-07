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

    @staticmethod
    def _build_react_messages(query: str, evidence: str) -> list:
        if evidence:
            return [
                {
                    "role": "system",
                    "content": "根据提供的检索结果回答。没有依据时只回答‘不知道’，不要编造。",
                },
                {
                    "role": "user",
                    "content": f"问题：{query}\n检索结果：\n{evidence}",
                },
            ]
        return [
            {
                "role": "system",
                "content": "如果无法从知识库中准确确认事实，请直接回答不知道，不要编造。",
            },
            {"role": "user", "content": query},
        ]

    def _call_llm(self, messages: list) -> str:
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.1,
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def run_react_query(self, query: str, initial_chunks: list | None = None) -> str:
        """兼容性降级：真正的 ReAct 决策循环已移交给 LangGraph。"""
        print(f"ReAct 兼容降级执行，问题: {query}")

        evidence = self._format_react_observation(initial_chunks) if initial_chunks else ""
        messages = self._build_react_messages(query, evidence)

        try:
            return self._call_llm(messages)
        except Exception as e:
            print(f"ReAct 兼容降级最终回答失败: {e}")
            return "服务响应失败，请稍后重试。"

    @staticmethod
    def _extract_react_query(decision: str) -> str:
        for marker in ("查询:", "查询：", "query:", "query："):
            if marker in decision:
                return decision.split(marker, 1)[1].splitlines()[0].strip()
        return ""

    @staticmethod
    def _extract_react_answer(decision: str) -> str:
        for marker in (
            "答案:",
            "答案：",
            "最终答案:",
            "最终答案：",
            "answer:",
            "answer：",
            "final answer:",
            "final answer：",
        ):
            if marker in decision:
                return decision.split(marker, 1)[1].strip()
        return decision if not decision.startswith(("行动", "action")) else ""

    @staticmethod
    def _format_react_observation(chunks: list) -> str:
        if not chunks:
            return "未检索到相关知识。"
        return "\n---\n".join(
            f"{index}. {chunk['content']}（相似度：{chunk['vector_sim']:.4f}）"
            for index, chunk in enumerate(chunks, 1)
        )

