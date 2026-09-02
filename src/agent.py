import requests
from src.config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    LLM_MODEL,
    VECTOR_SEARCH_THRESHOLD,
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
    def _build_grounded_system_prompt(context_text: str) -> str:
        return (
            "你是一个精通《最终幻想14》游戏机制的助手。请严格根据以下提供的背景知识回答玩家的问题。\n"
            "如果背景知识中没有提到相关信息，请直接回答不知道，绝对不要编造。\n\n"
            f"背景知识：\n{context_text}"
        )

    @staticmethod
    def _build_fallback_system_prompt() -> str:
        return (
            "你是一个精通《最终幻想14》游戏机制的助手。如果玩家提问的内容不是游戏相关的机制或你无法确定的内容，"
            "请直接回答不知道，绝对不要编造。"
        )

    @classmethod
    def _build_rag_messages(cls, query: str, chunks: list) -> list:
        if chunks:
            context_text = "\n---\n".join(chunk["content"] for chunk in chunks)
            system_prompt = cls._build_grounded_system_prompt(context_text)
        else:
            system_prompt = cls._build_fallback_system_prompt()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

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

    def run_query(self, query: str) -> str:
        print(f"智能体收到用户目标: {query}")

        retrieved_chunks = self.retriever.search(query, top_k=5)
        has_knowledge = False

        if retrieved_chunks:
            highest_vector_sim = float(max(chunk["vector_sim"] for chunk in retrieved_chunks))
            print(f"检索到候选片段，最高向量相似度为: {highest_vector_sim:.4f}")
            has_knowledge = highest_vector_sim >= float(VECTOR_SEARCH_THRESHOLD)
            if has_knowledge:
                print(f"最高向量相似度超过防幻觉阈值 {VECTOR_SEARCH_THRESHOLD}，采用本地知识库内容")
            else:
                print(f"最高向量相似度低于防幻觉阈值 {VECTOR_SEARCH_THRESHOLD}，判定为无相关本地知识")
        else:
            print("未检索到任何背景片段")

        messages = self._build_rag_messages(query, retrieved_chunks if has_knowledge else [])

        print("正在调用大模型生成最终回复...")
        try:
            return self._call_llm(messages)
        except Exception as e:
            print(f"大模型调用失败: {e}")
            return "服务响应失败，请稍后重试。"

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

if __name__ == "__main__":
    agent = GameAgent()
    test_query = "月读极神的核心机制是什么？"
    final_answer = agent.run_query(test_query)
    print(f"最终答案:\n {final_answer}")