try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None
    START = None
    END = None
    LANGGRAPH_AVAILABLE = False

from src.agent import GameAgent
from src.config import VECTOR_SEARCH_THRESHOLD, REACT_MAX_STEPS


class LangGraphAgent:
    """LangGraph 编排器：既保留传统 RAG 流程，也支持真正的 ReAct 图。"""

    def __init__(self):
        self.base_agent = GameAgent()
        self.retriever = self.base_agent.retriever
        self.graph = None
        self.react_graph = None
        self.max_steps = int(REACT_MAX_STEPS)

        self._build_standard_graph()
        self._build_react_graph()

    def _build_standard_graph(self):
        if not (LANGGRAPH_AVAILABLE and StateGraph is not None):
            self.graph = None
            return

        try:
            graph_builder = StateGraph(dict)

            def retrieve_node(state: dict):
                query = state.get("query", "")
                state["retrieved_chunks"] = self.retriever.search(query, top_k=5)
                return "decide"

            def decide_node(state: dict):
                chunks = state.get("retrieved_chunks") or []
                if chunks:
                    highest = float(max(c.get("vector_sim", 0.0) for c in chunks))
                    state["highest_vector_sim"] = highest
                    state["need_rag"] = highest >= float(VECTOR_SEARCH_THRESHOLD)
                else:
                    state["highest_vector_sim"] = 0.0
                    state["need_rag"] = False
                return "call_llm"

            def call_llm_node(state: dict):
                query = state.get("query", "")
                chunks = state.get("retrieved_chunks") if state.get("need_rag") else []
                context_text = "\n---\n".join([c.get("content", "") for c in chunks]) if chunks else ""
                if context_text:
                    system_prompt = (
                        "你是一个精通《最终幻想14》游戏机制的助手。请严格根据以下提供的背景知识回答玩家的问题。"
                        "如果背景知识中没有提到相关信息，请直接回答不知道，绝对不要编造。\n\n"
                        f"背景知识：\n{context_text}"
                    )
                else:
                    system_prompt = (
                        "你是一个精通《最终幻想14》游戏机制的助手。如果玩家提问的内容不是游戏相关的机制或你无法确定的内容，"
                        "请直接回答不知道，绝对不要编造。"
                    )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ]
                try:
                    state["answer"] = self.base_agent._call_llm(messages)
                except Exception as e:
                    state["answer"] = f"LLM 调用失败: {e}"
                return END

            graph_builder.add_node("retrieve", retrieve_node)
            graph_builder.add_node("decide", decide_node)
            graph_builder.add_node("call_llm", call_llm_node)

            graph_builder.add_edge(START, "retrieve")
            graph_builder.add_edge("retrieve", "decide")
            graph_builder.add_edge("decide", "call_llm")
            graph_builder.add_edge("call_llm", END)

            self.graph = graph_builder.compile()
        except Exception:
            self.graph = None

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

    def _build_react_prompt(self, query: str, messages: list, steps: int, observation: str = "") -> list:
        return [
            {"role": "system", "content": (
                "你是游戏知识助手。你可以使用一个工具：search_knowledge(query)。\n"
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
        if not (LANGGRAPH_AVAILABLE and StateGraph is not None):
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

            def search_node(state: dict):
                search_query = state.get("action_input") or state.get("question", "")
                chunks = self.retriever.search(search_query, top_k=5)
                observation = self._format_react_observation(chunks)
                state["retrieved_chunks"] = chunks
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
                    messages = [{
                        "role": "system",
                        "content": "根据提供的检索结果回答。没有依据时只回答‘不知道’，不要编造。",
                    }, {
                        "role": "user",
                        "content": f"问题：{query}\n检索结果：\n{evidence or '无'}",
                    }]
                    try:
                        answer = self.base_agent._call_llm(messages)
                    except Exception as e:
                        answer = f"服务响应失败，请稍后重试。原因: {e}"
                state["answer"] = answer
                return state

            def should_continue(state: dict):
                if state.get("next_action") == "search":
                    return "search"
                return "final"

            graph_builder.add_node("agent", agent_node)
            graph_builder.add_node("search", search_node)
            graph_builder.add_node("final", final_node)
            graph_builder.add_conditional_edges("agent", should_continue, {"search": "search", "final": "final"})
            graph_builder.add_edge("search", "agent")
            graph_builder.add_edge(START, "agent")
            graph_builder.add_edge("final", END)

            self.react_graph = graph_builder.compile()
        except Exception:
            self.react_graph = None

    def _invoke_graph(self, graph, state: dict):
        if graph is None:
            return None

        try:
            if hasattr(graph, "invoke"):
                return graph.invoke(state)
            if hasattr(graph, "run"):
                return graph.run(state)
            if hasattr(graph, "execute"):
                return graph.execute(state)
        except Exception:
            return None
        return None

    def run_query(self, query: str) -> str:
        """Execute the graph if available, otherwise fallback to sequential run."""
        if self.graph is not None:
            try:
                state = {"query": query}
                result = self._invoke_graph(self.graph, state)
                if isinstance(result, dict):
                    return result.get("answer")
                if isinstance(state, dict):
                    return state.get("answer")
            except Exception:
                pass

        return self._sequential_run(query)

    def _sequential_run(self, query: str) -> str:
        return self.base_agent.run_query(query)

    def run_react_query(self, query: str, initial_chunks: list | None = None) -> str:
        """真正的 ReAct 图执行入口。"""
        if self.react_graph is not None:
            initial_state = {
                "question": query,
                "messages": [{"role": "user", "content": query}],
                "steps": 0,
                "max_steps": self.max_steps,
                "observation": "",
                "retrieved_chunks": initial_chunks or [],
                "final_answer": "",
                "answer": "",
                "done": False,
            }
            result = self._invoke_graph(self.react_graph, initial_state)
            if isinstance(result, dict):
                answer = result.get("answer") or result.get("final_answer")
                if answer:
                    return answer

        return self.base_agent.run_react_query(query, initial_chunks)

    # 兼容性 shim：暴露 _call_llm 以匹配 GameAgent API
    def _call_llm(self, messages: list) -> str:
        if hasattr(self, "base_agent") and hasattr(self.base_agent, "_call_llm"):
            return self.base_agent._call_llm(messages)
        raise AttributeError("底层 agent 不包含 _call_llm 方法")
