import uuid

from src.data_preparation import DataPreparationPipeline, chunk_text


def test_chunk_text_keeps_content_intact():
    text = "第一句。第二句。第三句。第四句。"
    chunks = chunk_text(text, chunk_size=18, overlap=6)

    assert chunks
    assert all(len(chunk) <= 18 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_prepare_from_text_successfully_indexes_chunks(monkeypatch, tmp_path):
    collection_name = f"pytest_game_wiki_{uuid.uuid4().hex[:8]}"
    pipeline = DataPreparationPipeline(collection_name=collection_name)

    monkeypatch.setattr(pipeline, "_batch_embedding", lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
    monkeypatch.setattr(pipeline, "_delete_old_source", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_save_processed_text", lambda file_name, cleaned_text: str(tmp_path / file_name))

    result = pipeline.prepare_from_text("这是第一段内容。这里是第二段内容。", "demo_source")

    assert result["status"] == "success"
    assert result["chunk_count"] >= 1
    assert result["source_name"] == "demo_source"
