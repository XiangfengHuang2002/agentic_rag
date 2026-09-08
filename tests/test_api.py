import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.data_preparation import DataPreparationPipeline


@pytest.fixture
def client():
    return TestClient(app)


def test_chat_stream_and_prepare_progress_endpoints(client):
    class DummyAgent:
        def run_react_query(self, query, event_callback=None):
            if event_callback is not None:
                event_callback({"event": "node", "node": "agent", "message": "ok"})
            return "测试答案"

    app.state.agent = DummyAgent()
    app.state.data_pipeline = DataPreparationPipeline(collection_name="pytest_progress")

    stream_response = client.get("/api/chat/stream", params={"query": "测试问题"})

    assert stream_response.status_code == 200
    text = stream_response.text
    assert "event: status" in text or "event: result" in text
    assert "测试答案" in text or "测试问题" in text

    progress_response = client.get("/api/data/prepare/progress")

    assert progress_response.status_code == 200
    payload = progress_response.json()
    assert isinstance(payload, dict)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("", False),
        ("   ", False),
        ("新手入门", True),
    ],
)
def test_chat_stream_requires_nonempty_query(client, query, expected):
    response = client.get("/api/chat/stream", params={"query": query})

    if expected:
        assert response.status_code == 200
    else:
        assert response.status_code == 400
