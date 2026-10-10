import src.langgraph_orchestrator as orchestrator


class StubRetriever:
    def search(self, query, top_k=5):
        return [{"content": "低相似度候选证据", "vector_sim": 0.49}]


class StubAgent:
    def __init__(self):
        self.retriever = StubRetriever()
        self.calls = 0

    def _call_llm(self, messages):
        self.calls += 1
        if self.calls == 1:
            return "行动: search\n查询: 测试问题"
        return "行动: final\n答案: 不知道"


def test_react_threshold_blocks_low_score_evidence(monkeypatch):
    agent = StubAgent()
    monkeypatch.setattr(orchestrator, "GameAgent", lambda: agent)
    monkeypatch.setattr(orchestrator, "VECTOR_SEARCH_THRESHOLD", 0.5)
    events = []

    graph_agent = orchestrator.LangGraphAgent()
    result = graph_agent._invoke_graph(
        graph_agent.react_graph,
        {
            "question": "测试问题",
            "messages": [{"role": "user", "content": "测试问题"}],
            "steps": 0,
            "observation": "",
            "retrieved_chunks": [],
            "final_answer": "",
            "answer": "",
            "event_callback": events.append,
        },
    )

    assert result["answer"] == "不知道"
    assert result["rag_selected"] is False
    assert result["retrieved_chunks"] == []
    assert result["observation"] == "未检索到相关知识。"
    rag_event = next(event for event in events if event.get("node") == "rag")
    assert rag_event["candidate_count"] == 1
    assert rag_event["evidence_count"] == 0
    assert rag_event["chunks"] == []
    assert rag_event["threshold"] == 0.5
    action_event = next(
        event for event in events
        if event.get("event") == "decision" and "score" not in event
    )
    assert action_event["tool_requested"] is True
    assert "need_rag" not in action_event
    gate_event = next(
        event for event in events
        if event.get("event") == "decision" and "score" in event
    )
    assert gate_event["score"] == 0.49
    assert gate_event["need_rag"] is False


def test_react_prompt_routes_general_concepts_away_from_game_retrieval():
    prompt = orchestrator.LangGraphAgent._build_react_prompt(
        "什么是 Agentic RAG？", [], 0
    )

    assert "不要检索 FF14 知识库" in prompt[0]["content"]
    assert "Agentic RAG" in prompt[0]["content"]