try:
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    StateGraph = None
    START = None
    END = None
    LANGGRAPH_AVAILABLE = False

from src.agent import GameAgent
from src.config import VECTOR_SEARCH_THRESHOLD


class LangGraphAgent:
    """LangGraph 简易编排器包装。

    - 当可导入 `langgraph` 时，构建一个最小的状态图描述流程（retrieve -> decide -> call_llm）。
      由于不同版本的编译图可能在运行时接口上有所差异，`run_query` 在必要时会回退到顺序执行。
    - 若不可用，则委托给原始的 `GameAgent` 实现。
    """

    def __init__(self):
        self.base_agent = GameAgent()
        self.retriever = self.base_agent.retriever
        self.graph = None

        if LANGGRAPH_AVAILABLE and StateGraph is not None:
            try:
                # 构建描述步骤的最小 StateGraph。
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
                            "你是一个精通游戏机制的助手。请严格根据以下提供的背景知识回答玩家的问题。"
                            "如果背景知识中没有提到相关信息，请直接回答不知道，绝对不要编造。\n\n"
                            f"背景知识：\n{context_text}"
                        )
                    else:
                        system_prompt = (
                            "你是一个精通游戏机制的助手。如果玩家提问的内容不是游戏相关的机制或你无法确定的内容，"
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

                # 若支持则尝试编译
                try:
                    self.graph = graph_builder.compile()
                except Exception:
                    self.graph = None
            except Exception:
                self.graph = None

    def run_query(self, query: str) -> str:
        """Execute the graph if available, otherwise fallback to sequential run."""
        if self.graph is not None:
            try:
                state = {"query": query}
                # 尝试常见的运行入口，若未知则回退到顺序执行。
                if hasattr(self.graph, "run"):
                    result = self.graph.run(state)
                elif hasattr(self.graph, "execute"):
                    result = self.graph.execute(state)
                else:
                    # 编译后的图接口未知，执行顺序回退
                    return self._sequential_run(query)

                # result may be dict-like or the mutated state
                if isinstance(result, dict):
                    return result.get("answer")
                if isinstance(state, dict):
                    return state.get("answer")
                return None
            except Exception:
                return self._sequential_run(query)
        else:
            return self._sequential_run(query)

    def _sequential_run(self, query: str) -> str:
        return self.base_agent.run_query(query)

    def run_react_query(self, query: str, initial_chunks: list | None = None) -> str:
        """转发 ReAct 执行入口。"""
        return self.base_agent.run_react_query(query, initial_chunks)

    # 兼容性 shim：暴露 _call_llm 以匹配 GameAgent API
    def _call_llm(self, messages: list) -> str:
        if hasattr(self, "base_agent") and hasattr(self.base_agent, "_call_llm"):
            return self.base_agent._call_llm(messages)
        raise AttributeError("底层 agent 不包含 _call_llm 方法")
