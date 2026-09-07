from langgraph.graph import StateGraph, START, END

from src.agent import GameAgent
from src.config import VECTOR_SEARCH_THRESHOLD, REACT_MAX_STEPS


class LangGraphAgent:
    """ReAct 编排器：RAG 只作为图中的检索节点。"""

    def __init__(self):
        self.base_agent = GameAgent()
        self.retriever = self.base_agent.retriever
        self.react_graph = None
        self.max_steps = int(REACT_MAX_STEPS)

        self._build_react_graph()

    def run_react_query(self, query: str) -> str:
        """执行 ReAct 图并返回最终答案。"""
        if self.react_graph is None:
            raise RuntimeError("ReAct 图初始化失败")

        result = self._invoke_graph(
            self.react_graph,
            {
                "question": query,
                "messages": [{"role": "user", "content": query}],
                "steps": 0,
                "observation": "",
                "retrieved_chunks": [],
                "final_answer": "",
                "answer": "",
            },
        )
        if isinstance(result, dict):
            answer = result.get("answer") or result.get("final_answer")
            if answer:
                return answer
        raise RuntimeError("ReAct 图未生成最终答案")

    @staticmethod
    def _format_react_observation(chunks: list) -> str:
        if not chunks:
            return "未检索到相关知识。"
        return "\n---\n".join(
            f"{index}. {chunk['content']}（相似度：{chunk['vector_sim']:.4f}）"
            for index, chunk in enumerate(chunks, 1)
        )

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
    def _build_react_prompt(query: str, messages: list, steps: int, observation: str = "") -> list:
        return [
            {"role": "system", "content": (
                "你是《最终幻想14》游戏知识助手。你可以使用一个工具：search_knowledge(query)。\n"
                "每轮只能输出一行：行动: search\n查询: 你的检索词，或行动: final\n答案: 你的最终答案。\n"
                "需要事实依据时先检索；已有足够依据时直接回答。不要编造知识。"
                f"当前已执行步数: {steps}。"
                f"\n上一步观察：\n{observation or '无'}"
            )},
            {"role": "user", "content": query},
        ] + messages

    def _parse_react_decision(self, decision: str, query: str) -> dict:
        text = decision.strip()
        normalized = text.lower().replace(" ", "")

        if (
            "行动:search" in normalized
            or "行动：search" in normalized
            or "action:search" in normalized
            or "action：search" in normalized
        ):
            action_query = self._extract_react_query(text) or query
            return {
                "next_action": "search",
                "action_input": action_query,
                "decision": text,
                "final_answer": "",
            }

        answer = self._extract_react_answer(text)
        if answer:
            return {
                "next_action": "final",
                "action_input": "",
                "decision": text,
                "final_answer": answer,
            }

        return {
            "next_action": "final",
            "action_input": "",
            "decision": text,
            "final_answer": "",
        }

    def _build_react_graph(self):
        if not (StateGraph is not None):
            self.react_graph = None
            return

        try:
            graph_builder = StateGraph(dict)

            def agent_node(state: dict):
                query = state.get("question", "")
                steps = int(state.get("steps", 0))
                messages = list(state.get("messages", []))
                observation = state.get("observation", "")
                prompt = self._build_react_prompt(query, messages, steps, observation)
                try:
                    decision = self.base_agent._call_llm(prompt).strip()
                except Exception as e:
                    decision = f"行动: final\n答案: 服务响应失败，请稍后重试。原因: {e}"

                parsed = self._parse_react_decision(decision, query)
                state["decision"] = parsed["decision"]
                state["next_action"] = parsed["next_action"]
                state["action_input"] = parsed["action_input"]
                state["final_answer"] = parsed["final_answer"]
                state["steps"] = steps + 1
                state["messages"] = messages + [{"role": "assistant", "content": decision}]
                return state

            def rag_node(state: dict):
                search_query = state.get("action_input") or state.get("question", "")
                chunks = self.retriever.search(search_query, top_k=5)
                highest_score = max(
                    (float(chunk.get("vector_sim", 0.0)) for chunk in chunks),
                    default=0.0,
                )
                observation = self._format_react_observation(chunks)
                state["retrieved_chunks"] = chunks
                state["highest_vector_sim"] = highest_score
                state["rag_selected"] = highest_score >= float(VECTOR_SEARCH_THRESHOLD)
                state["observation"] = observation
                state["messages"] = list(state.get("messages", [])) + [
                    {"role": "user", "content": f"观察结果：\n{observation}\n请继续决定行动。"}
                ]
                return state

            def final_node(state: dict):
                query = state.get("question", "")
                evidence = state.get("observation", "")
                answer = state.get("final_answer")
                if not answer:
                    messages = self._build_final_messages(query, evidence)
                    try:
                        answer = self.base_agent._call_llm(messages)
                    except Exception as e:
                        answer = f"服务响应失败，请稍后重试。原因: {e}"
                state["answer"] = answer
                return state

            def should_continue(state: dict):
                if (
                    state.get("next_action") == "search"
                    and int(state.get("steps", 0)) < self.max_steps
                ):
                    return "search"
                return "final"

            graph_builder.add_node("agent", agent_node)
            graph_builder.add_node("rag", rag_node)
            graph_builder.add_node("final", final_node)
            graph_builder.add_conditional_edges("agent", should_continue, {"search": "rag", "final": "final"})
            graph_builder.add_edge("rag", "agent")
            graph_builder.add_edge(START, "agent")
            graph_builder.add_edge("final", END)

            self.react_graph = graph_builder.compile()
        except Exception:
            self.react_graph = None

    @staticmethod
    def _build_final_messages(query: str, evidence: str) -> list:
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

    def _invoke_graph(self, graph, state: dict):
        if graph is None:
            raise RuntimeError("ReAct 图未初始化")
        return graph.invoke(state)

