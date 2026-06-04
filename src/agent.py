import os
import requests
from src.config import SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL, LLM_MODEL, VECTOR_SEARCH_THRESHOLD
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

if __name__ == "__main__":
    agent = GameAgent()
    test_query = "月读极神的核心机制是什么？"
    final_answer = agent.run_query(test_query)
    print(f"最终答案:\n {final_answer}")