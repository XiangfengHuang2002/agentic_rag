from langgraph.graph import StateGraph, START, END

from src.agent import GameAgent
from src.config import VECTOR_SEARCH_THRESHOLD, REACT_MAX_STEPS


class LangGraphAgent:
    """ReAct 编排器：RAG 只作为图中的检索节点。"""

    def __init__(self):
        """初始化底层 Agent、检索器和 ReAct LangGraph。

        输入：无；最大推理步数读取自配置项 `REACT_MAX_STEPS`。
        输出：无；初始化结果保存在 `react_graph` 属性中。
        """
        self.base_agent = GameAgent()
        self.retriever = self.base_agent.retriever
        self.max_steps = int(REACT_MAX_STEPS)

        self._build_react_graph()

    def run_react_query(self, query: str) -> str:
        """执行 ReAct 图并返回最终回答。

        输入：`query`，用户的自然语言问题。
        输出：ReAct 图生成的最终回答字符串。
        异常：图未生成回答时抛出 `RuntimeError`；LangGraph 执行错误直接向上传递。
        """
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
        """把检索片段格式化为供下一轮 ReAct 决策读取的观察文本。

        输入：`chunks`，包含 `content` 和 `vector_sim` 字段的检索结果列表。
        输出：带序号和相似度的观察字符串；列表为空时返回未命中提示。
        """
        if not chunks:
            return "未检索到相关知识。"
        return "\n---\n".join(
            f"{index}. {chunk['content']}（相似度：{chunk['vector_sim']:.4f}）"
            for index, chunk in enumerate(chunks, 1)
        )

    @staticmethod
    def _extract_react_query(decision: str) -> str:
        """从模型的 search 决策文本中提取下一次检索词。

        输入：`decision`，模型输出的行动文本。
        输出：`查询:` 或 `query:` 后面的单行检索词；未找到时返回空字符串。
        """
        for marker in ("查询:", "查询：", "query:", "query："):
            if marker in decision:
                return decision.split(marker, 1)[1].splitlines()[0].strip()
        return ""

    @staticmethod
    def _extract_react_answer(decision: str) -> str:
        """从模型的 final 决策文本中提取最终答案。

        输入：`decision`，模型输出的行动文本。
        输出：答案标记后的文本；无法提取时返回原文本或空字符串。
        """
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
    def _build_react_prompt(
        query: str, messages: list, steps: int, observation: str = ""
    ) -> list:
        """构造一次 ReAct 决策所需的系统提示、用户问题和历史消息。

        输入：`query` 为原始问题，`messages` 为历史消息，`steps` 为已执行步数，`observation` 为上一轮观察。
        输出：可直接发送给对话模型的消息列表。
        """
        return [
            {
                "role": "system",
                "content": (
                    "你是《最终幻想14》游戏知识助手。你可以使用一个工具：search_knowledge(query)。\n"
                    "每轮只能输出一行：行动: search\n查询: 你的检索词，或行动: final\n答案: 你的最终答案。\n"
                    "需要事实依据时先检索；已有足够依据时直接回答。不要编造知识。"
                    f"当前已执行步数: {steps}。"
                    f"\n上一步观察：\n{observation or '无'}"
                ),
            },
            {"role": "user", "content": query},
        ] + messages

    def _parse_react_decision(self, decision: str, query: str) -> dict:
        """把模型文本解析成图可执行的结构化行动。

        输入：`decision` 为模型输出文本，`query` 为原始问题，用于缺少检索词时兜底。
        输出：包含 `next_action`、`action_input`、`decision` 和 `final_answer` 的状态字典。
        """
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
        """创建并编译 ReAct 图及其 agent、rag、final 节点。

        输入：无；使用实例中的检索器、底层 Agent 和最大步数配置。
        输出：无；编译后的图写入 `self.react_graph`。
        """
        graph_builder = StateGraph(dict)

        def agent_node(state: dict):
            """调用模型决定继续检索还是生成最终答案。

            输入：包含问题、历史消息和观察结果的图状态字典。
            输出：更新行动、行动参数、答案和步数后的状态字典。
            """
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
            state["messages"] = messages + [
                {"role": "assistant", "content": decision}
            ]
            return state

        def rag_node(state: dict):
            """执行知识库检索，并把结果写入 ReAct 观察状态。

            输入：包含 `action_input` 或 `question` 的图状态字典。
            输出：补充检索片段、最高相似度、RAG 选择结果和观察文本的状态字典。
            """
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
                {
                    "role": "user",
                    "content": f"观察结果：\n{observation}\n请继续决定行动。",
                }
            ]
            return state

        def final_node(state: dict):
            """根据已有行动答案或检索观察生成最终回答。

            输入：包含问题、观察文本和可选 `final_answer` 的图状态字典。
            输出：补充 `answer` 字段后的状态字典。
            """
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
            """根据模型行动和步数上限选择下一个图节点。

            输入：包含 `next_action` 和 `steps` 的图状态字典。
            输出：`search` 或 `final`，分别映射到 `rag` 节点或最终节点。
            """
            if (
                state.get("next_action") == "search"
                and int(state.get("steps", 0)) < self.max_steps
            ):
                return "search"
            return "final"

        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("rag", rag_node)
        graph_builder.add_node("final", final_node)
        graph_builder.add_conditional_edges(
            "agent", should_continue, {"search": "rag", "final": "final"}
        )
        graph_builder.add_edge("rag", "agent")
        graph_builder.add_edge(START, "agent")
        graph_builder.add_edge("final", END)

        self.react_graph = graph_builder.compile()

    @staticmethod
    def _build_final_messages(query: str, evidence: str) -> list:
        """构造最终回答阶段使用的消息列表。

        输入：`query` 为用户问题，`evidence` 为 RAG 节点生成的观察文本。
        输出：包含系统约束和用户问题的对话消息列表。
        """
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
        """调用已编译的 LangGraph 并返回执行后的状态。

        输入：`graph` 为已编译图，`state` 为初始图状态字典。
        输出：LangGraph 返回的最终状态字典。
        异常：LangGraph 执行异常直接向上传递。
        """
        return graph.invoke(state)
