import html
import json
import os
import re
import socket
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import chromadb
import requests

from src.config import CHROMA_DB_DIR, EMBEDDING_MODEL, SILICONFLOW_API_KEY, SILICONFLOW_BASE_URL

MAX_EMBEDDING_CHARS = 1200

# ===== 强制使用 IPv4 解决 Windows DNS 解析问题 =====
import urllib3.util.connection as urllib3_cn

def allowed_gateways():
    return socket.AF_INET  # 强制只使用 IPv4

urllib3_cn.allowed_gateways = allowed_gateways
# ================================================


def clean_wiki_text(raw_text: str) -> str:
    """清洗 Wiki/HTML 富文本，保留可读正文。"""
    if raw_text is None:
        return ""

    text = str(raw_text)
    text = html.unescape(text)

    replacements = [
        (r"<br\s*/?>", "\n", False),
        (r"<\s*/?\s*(div|p|blockquote|li|ul|ol|tr|td|th|table|span|section|article|b|strong|i|em|code|pre)\s*>", " ", False),
        (r"<[^>]+>", " ", False),
        (r"\{\{.*?\}\}", lambda m: _wiki_template_to_text(m.group(0)), True),
        (r"\[\[(.*?)\]\]", lambda m: _wiki_link_to_text(m.group(1)), True),
        (r"\[[^\]]*\]\s*", " ", False),
        (r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", False),
        (r"\[https?://[^\]]+\]", " ", False),
        (r"^==+\s*(.*?)\s*==+\s*$", r"\1\n", True),
        (r"^===\s*(.*?)\s*===\s*$", r"\1\n", True),
    ]

    for pattern, repl, use_callable in replacements:
        if use_callable:
            text = re.sub(pattern, repl, text, flags=re.S)
        else:
            text = re.sub(pattern, repl, text, flags=re.S | re.I)

    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+\n\s+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u200b", "")
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _wiki_template_to_text(template: str) -> str:
    inner = template[2:-2].strip()
    if "|" in inner:
        parts = [p.strip() for p in inner.split("|") if p.strip()]
        return parts[-1] if parts else ""
    return inner


def _wiki_link_to_text(link: str) -> str:
    content = link.strip()
    if "|" in content:
        return content.split("|")[-1]
    if ":" in content and "/" not in content:
        return content.split(":", 1)[-1]
    return content


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 60) -> List[str]:
    """将完整文本按句子边界切块，必要时使用滑动窗口，保证不丢字符。"""
    if not text or not text.strip():
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size 必须大于 0，overlap 必须小于 chunk_size")
    chunk_size = min(chunk_size, MAX_EMBEDDING_CHARS)
    overlap = min(overlap, chunk_size - 1)

    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        if end < len(cleaned):
            # 在窗口尾部寻找句号、换行等自然边界；找不到时硬切，仍保证覆盖完整。
            boundary_start = start + max(1, chunk_size // 2)
            boundary = max(
                cleaned.rfind(mark, boundary_start, end)
                for mark in ("。", "！", "？", "!", "?", ";", "；", ",", "，", " ")
            )
            if boundary > start:
                end = boundary + 1 if cleaned[boundary] != " " else boundary

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        next_start = max(start + 1, end - overlap)
        start = next_start

    return chunks


class DataPreparationPipeline:
    """统一的数据准备 pipeline：清洗 -> 切块 -> embedding -> 写入 ChromaDB。"""

    def __init__(self, collection_name: str = "game_wiki"):
        self.collection_name = collection_name
        self.progress: Dict[str, Any] = {
            "status": "idle",
            "stage": "未开始",
            "message": "等待导入文本或文件",
            "percent": 0,
            "source_name": "",
            "chunk_count": 0,
            "total_chars": 0,
        }
        self.current_source: str = ""
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _set_progress(self, stage: str, message: str, percent: int, source_name: str | None = None, chunk_count: int = 0, total_chars: int = 0):
        self.progress.update({
            "status": "running" if percent < 100 else "completed",
            "stage": stage,
            "message": message,
            "percent": max(0, min(100, percent)),
            "source_name": source_name or self.current_source,
            "chunk_count": chunk_count,
            "total_chars": total_chars,
        })

    def _embedding_for_text(self, text: str) -> List[float]:
        text = str(text).strip()
        if not text:
            raise ValueError("embedding 文本不能为空")
        if len(text) > MAX_EMBEDDING_CHARS:
            raise ValueError(f"单个 embedding chunk 不能超过 {MAX_EMBEDDING_CHARS} 字符，请先分块")
        if not SILICONFLOW_API_KEY:
            raise RuntimeError("未配置 SILICONFLOW_API_KEY，无法生成 embedding。")

        url = f"{SILICONFLOW_BASE_URL}/embeddings"
        payload = {
            "model": EMBEDDING_MODEL,
            "input": text,
            "encoding_format": "float",
        }
        response = requests.post(url, headers={"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}, json=payload, timeout=60)
        response.raise_for_status()
        item = response.json()["data"][0]
        return item["embedding"]

    def _save_processed_text(self, file_name: str, cleaned_text: str):
        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = output_dir / file_name
        target_path.write_text(cleaned_text, encoding="utf-8")
        return str(target_path)

    def prepare_from_text(self, raw_text: str, source_name: str = "uploaded_text") -> Dict[str, Any]:
        source_name = (source_name or "uploaded_text").strip() or "uploaded_text"
        self.current_source = source_name
        self.progress.update({
            "status": "running",
            "stage": "received",
            "message": "已接收源文本，开始清洗...",
            "percent": 5,
            "source_name": source_name,
        })

        if raw_text is None or not str(raw_text).strip():
            raise ValueError("源文本不能为空")

        source_text = str(raw_text)
        self.progress.update({
            "message": "已接收完整源文本，开始清洗...",
            "truncated": False,
            "input_chars": len(str(raw_text)),
        })

        cleaned_text = clean_wiki_text(source_text)
        self._set_progress("cleaning", "正在清洗 Wiki/HTML 标记并去除噪声...", 30, source_name, total_chars=len(cleaned_text))

        chunks = chunk_text(cleaned_text)
        self._set_progress("chunking", "文本已切分为多个知识块，准备生成 embedding...", 60, source_name, chunk_count=len(chunks), total_chars=len(cleaned_text))

        if not chunks:
            raise ValueError("清洗后没有可入库的文本内容")

        embeddings: List[List[float]] = []
        for index, chunk in enumerate(chunks, 1):
            embeddings.append(self._embedding_for_text(chunk))
            progress = int(60 + (index / max(1, len(chunks))) * 30)
            self._set_progress("embedding", f"正在生成 embedding（{index}/{len(chunks)}）...", progress, source_name, chunk_count=len(chunks), total_chars=len(cleaned_text))

        if not embeddings:
            raise RuntimeError("未生成任何 embedding")

        ids = [f"{source_name}-{uuid4()}" for _ in chunks]
        metadatas = [{"source": source_name, "chunk_index": index, "length": len(chunk)} for index, chunk in enumerate(chunks)]
        self._set_progress("indexing", "正在将知识块写入 ChromaDB...", 95, source_name, chunk_count=len(chunks), total_chars=len(cleaned_text))
        self.collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

        processed_path = self._save_processed_text(f"{source_name}.txt", cleaned_text)
        self._set_progress("indexing", "清洗后的文本和 embedding 已写入 ChromaDB，流程完成。", 100, source_name, chunk_count=len(chunks), total_chars=len(cleaned_text))

        return {
            "status": "success",
            "source_name": source_name,
            "processed_path": processed_path,
            "chunk_count": len(chunks),
            "total_chars": len(cleaned_text),
            "progress": self.progress.copy(),
        }

    def prepare_from_raw_dir(self, raw_dir: str = "data/raw") -> Dict[str, Any]:
        raw_path = Path(raw_dir)
        if not raw_path.exists():
            raise FileNotFoundError(f"找不到原始文本目录：{raw_dir}")

        summaries: List[Dict[str, Any]] = []
        for file_path in sorted(raw_path.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            result = self.prepare_from_text(text, source_name=file_path.stem)
            summaries.append(result)

        return {
            "status": "success",
            "source_count": len(summaries),
            "items": summaries,
        }


def process_raw_directory(raw_dir: str = "data/raw") -> Dict[str, Any]:
    pipeline = DataPreparationPipeline()
    return pipeline.prepare_from_raw_dir(raw_dir)
