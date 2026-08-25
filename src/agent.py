import os
import requests
from src.config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    LLM_MODEL,
    VECTOR_SEARCH_THRESHOLD,
    REACT_MAX_STEPS,
)
from src.retriever import WikiRetriever

class GameAgent:
    def __init__(self):
        self.retriever = WikiRetriever()
        self.url = f"{SILICONFLOW_BASE_URL}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }

    def _call_llm(self, messages: list) -> str:
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.1
        }
        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def run_query(self, query: str) -> str:
        print(f"智能体收到用户目标: {query}")
        
        retrieved_chunks = self.retriever.search(query, top_k=5)
        
        has_knowledge = False
        context_text = ""
        
        if retrieved_chunks:
            highest_vector_sim = float(max(chunk["vector_sim"] for chunk in retrieved_chunks))
            print(f"检索到候选片段，最高向量相似度为: {highest_vector_sim:.4f}")
            
            if highest_vector_sim >= float(VECTOR_SEARCH_THRESHOLD):
                print(f"最高向量相似度超过防幻觉阈值 {VECTOR_SEARCH_THRESHOLD}，采用本地知识库内容")
                has_knowledge = True
                context_text = "\n---\n".join([chunk["content"] for chunk in retrieved_chunks])
            else:
                print(f"最高向量相似度低于防幻觉阈值 {VECTOR_SEARCH_THRESHOLD}，判定为无相关本地知识")
        else:
            print("未检索到任何背景片段")

        if has_knowledge:
            system_prompt = (
                "你是一个精通游戏机制的助手。请严格根据以下提供的背景知识回答玩家的问题。\n"
                "如果背景知识中没有提到相关信息，请直接回答不知道，绝对不要编造。\n\n"
                f"背景知识：\n{context_text}"
            )
        else:
            system_prompt = "你是一个精通游戏机制的助手。如果玩家提问的内容不是游戏相关的机制或你无法确定的内容，请直接回答不知道，绝对不要编造。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        print("正在调用大模型生成最终回复...")
        try:
            answer = self._call_llm(messages)
            return answer
        except Exception as e:
            print(f"大模型调用失败: {e}")
            return "服务响应失败，请稍后重试。"

    def run_react_query(self, query: str, initial_chunks: list | None = None) -> str:
        """执行 ReAct 循环：行动、观察，再回答。"""
        print(f"ReAct 智能体收到问题: {query}")
        messages = [{
            "role": "system",
            "content": (
                "你是游戏知识助手。你可以使用一个工具：search_knowledge(query)，用于检索本地知识库。\n"
                "每轮只能输出一行：行动: search\n查询: 你的检索词，或行动: final\n答案: 你的最终答案。\n"
                "需要事实依据时先检索；已有足够依据时直接回答。不要编造知识。"
            ),
        }, {"role": "user", "content": query}]
        observations = []

        if initial_chunks is not None:
            observation = self._format_react_observation(initial_chunks)
            observations.append(observation)
            messages.append({
                "role": "user",
                "content": f"初始检索结果：\n{observation}\n请根据结果决定下一步行动。",
            })

        for step in range(max(1, REACT_MAX_STEPS)):
            try:
                decision = self._call_llm(messages).strip()
            except Exception as e:
                print(f"ReAct 第 {step + 1} 步调用失败: {e}")
                return "服务响应失败，请稍后重试。"

            normalized_decision = decision.lower().replace(" ", "")
            if (
                "行动:search" in normalized_decision
                or "行动：search" in normalized_decision
                or "action:search" in normalized_decision
                or "action：search" in normalized_decision
            ):
                search_query = self._extract_react_query(decision) or query
                chunks = self.retriever.search(search_query, top_k=5)
                observation = self._format_react_observation(chunks)
                observations.append(observation)
                messages.extend([
                    {"role": "assistant", "content": decision},
                    {"role": "user", "content": f"观察结果：\n{observation}\n请继续决定行动。"},
                ])
                continue

            answer = self._extract_react_answer(decision)
            if answer:
                return answer

            messages.extend([
                {"role": "assistant", "content": decision},
                {"role": "user", "content": "请按规定格式输出行动或最终答案。"},
            ])

        evidence = "\n\n".join(observations)
        final_messages = [{
            "role": "system",
            "content": "根据提供的检索结果回答。没有依据时只回答‘不知道’，不要编造。",
        }, {
            "role": "user",
            "content": f"问题：{query}\n检索结果：\n{evidence or '无'}",
        }]
        try:
            return self._call_llm(final_messages)
        except Exception as e:
            print(f"ReAct 最终回答失败: {e}")
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