import html
import hashlib
import re
import socket
import time
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import requests
from src.mediawiki_parser import parse_mediawiki

from src.config import (
    CHROMA_DB_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL,
    MAX_EMBEDDING_CHARS,
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
)

# ===== 强制使用 IPv4 解决 Windows DNS 解析问题 =====
import urllib3.util.connection as urllib3_cn

def allowed_gateways():
    """返回网络请求使用的地址族。

    输入：无。
    输出：`socket.AF_INET`，表示强制使用 IPv4。
    """
    return socket.AF_INET  # 强制只使用 IPv4

urllib3_cn.allowed_gateways = allowed_gateways
# ================================================


def clean_wiki_text(raw_text: str) -> str:
    """清洗 Wiki/HTML 富文本并保留可读正文。

    输入：`raw_text`，原始 Wiki、HTML 或纯文本内容。
    输出：去除标记、规范空白并保留链接文字后的纯文本；输入为 `None` 时返回空字符串。
    """
    if raw_text is None:
        return ""

    return parse_mediawiki(str(raw_text)).plain_text


def _wiki_template_to_text(template: str) -> str:
    """提取 Wiki 模板中的可读文本或最后一个参数。

    输入：`template`，包含 `{{...}}` 的模板字符串。
    输出：模板参数中最适合作为正文的文本字符串。
    """
    inner = template[2:-2].strip()
    if "|" in inner:
        parts = [p.strip() for p in inner.split("|") if p.strip()]
        return parts[-1] if parts else ""
    return inner


def _wiki_link_to_text(link: str) -> str:
    """把 Wiki 链接语法转换为用户可读的链接标题。

    输入：`link`，不含外层 `[[ ]]` 的 Wiki 链接内容。
    输出：链接标题、去除命名空间后的名称或原始内容。
    """
    content = link.strip()
    if "|" in content:
        return content.split("|")[-1]
    if ":" in content and "/" not in content:
        return content.split(":", 1)[-1]
    return content


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 60) -> List[str]:
    """按自然句子边界和滑动窗口切分文本，保证不丢失尾部内容。

    输入：`text`，待切分文本；`chunk_size`，单块最大字符数；`overlap`，相邻块重叠字符数。
    输出：文本块字符串列表；空文本返回空列表。
    异常：参数不满足窗口约束时抛出 `ValueError`。
    """
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
        """初始化数据准备进度、ChromaDB 客户端和目标集合。

        输入：`collection_name`，要写入或创建的 ChromaDB 集合名。
        输出：无；实例保存进度状态和数据库集合句柄。
        """
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
        """更新当前数据准备任务的进度字典。

        输入：阶段名、用户可读消息、百分比，以及可选来源名、块数和字符数。
        输出：无；修改实例的 `progress` 属性。
        """
        self.progress.update({
            "status": "running" if percent < 100 else "completed",
            "stage": stage,
            "message": message,
            "percent": max(0, min(100, percent)),
            "source_name": source_name or self.current_source,
            "chunk_count": chunk_count,
            "total_chars": total_chars,
        })

    def _batch_embedding(self, texts: List[str]) -> List[List[float]]:
        """批量调用 Embedding API；只对临时网络/5xx/429 错误重试。"""
        if not texts:
            return []
        if EMBEDDING_BATCH_SIZE <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE 必须大于 0")
        normalized = [str(text).strip() for text in texts]
        if any(not text for text in normalized):
            raise ValueError("embedding 文本不能为空")
        if any(len(text) > MAX_EMBEDDING_CHARS for text in normalized):
            raise ValueError(f"单个 embedding chunk 不能超过 {MAX_EMBEDDING_CHARS} 字符，请先分块")
        if not SILICONFLOW_API_KEY:
            raise RuntimeError("未配置 SILICONFLOW_API_KEY，无法生成 embedding。")

        url = f"{SILICONFLOW_BASE_URL}/embeddings"
        headers = {"Authorization": f"Bearer {SILICONFLOW_API_KEY}", "Content-Type": "application/json"}
        payload = {"model": EMBEDDING_MODEL, "input": normalized, "encoding_format": "float"}
        last_error = None
        for attempt in range(EMBEDDING_MAX_RETRIES + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                items = sorted(response.json()["data"], key=lambda item: item.get("index", 0))
                embeddings = [item["embedding"] for item in items]
                if len(embeddings) != len(normalized):
                    raise RuntimeError("embedding 返回数量与请求文本数量不一致")
                return embeddings
            except (requests.RequestException, KeyError, IndexError, RuntimeError) as exc:
                last_error = exc
                if attempt >= EMBEDDING_MAX_RETRIES:
                    break
                time.sleep(2 ** attempt)
        raise RuntimeError(f"批量 embedding 失败，已重试 {EMBEDDING_MAX_RETRIES} 次: {last_error}") from last_error

    def _embedding_for_text(self, text: str) -> List[float]:
        """兼容单条调用，实际复用批量 embedding 实现。"""
        return self._batch_embedding([text])[0]

    def _source_hash(self, cleaned_text: str) -> str:
        return hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()

    def _existing_chunks(self, source_name: str, source_hash: str) -> Dict[int, str]:
        """返回同一来源、同一文本版本已经入库的 chunk ID。"""
        results = self.collection.get(
            where={"$and": [{"source": source_name}, {"source_hash": source_hash}]},
            include=["metadatas"],
        )
        existing: Dict[int, str] = {}
        for chunk_id, metadata in zip(results.get("ids", []), results.get("metadatas", [])):
            if metadata and metadata.get("chunk_index") is not None:
                existing[int(metadata["chunk_index"])] = chunk_id
        return existing

    def _delete_old_source(self, source_name: str, source_hash: str):
        """来源内容更新时删除旧版本，避免检索同时命中过期数据。"""
        old = self.collection.get(where={"source": source_name}, include=["metadatas"])
        old_ids = [
            chunk_id for chunk_id, metadata in zip(old.get("ids", []), old.get("metadatas", []))
            if not metadata or metadata.get("source_hash") != source_hash
        ]
        if old_ids:
            self.collection.delete(ids=old_ids)

    def _save_processed_text(self, file_name: str, cleaned_text: str):
        """将清洗后的文本保存到 data 目录。

        输入：`file_name`，输出文件名；`cleaned_text`，清洗后的文本。
        输出：写入文件的字符串路径。
        """
        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = output_dir / file_name
        target_path.write_text(cleaned_text, encoding="utf-8")
        return str(target_path)

    def prepare_from_text(self, raw_text: str, source_name: str = "uploaded_text") -> Dict[str, Any]:
        """完成单个文本源的清洗、切块、向量化和 ChromaDB 入库。

        输入：`raw_text`，原始文本；`source_name`，用于元数据、文件名和任务进度的来源名。
        输出：包含状态、来源、处理文件、块数、字符数和最终进度的结果字典。
        异常：源文本为空、清洗后无内容或 embedding/入库失败时抛出异常。
        """
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

        parsed_document = parse_mediawiki(source_text)
        cleaned_text = parsed_document.plain_text
        template_summary = {
            name: len(records)
            for name, records in parsed_document.templates_by_name.items()
        }
        self.progress.update({
            "heading_count": len(parsed_document.headings),
            "link_count": len(parsed_document.links),
            "template_count": len(parsed_document.templates),
            "templates_by_name": template_summary,
        })
        source_hash = self._source_hash(cleaned_text)
        existing = self._existing_chunks(source_name, source_hash)
        if existing:
            expected_chunk_count = len(chunk_text(cleaned_text))
            if len(existing) == expected_chunk_count:
                self._set_progress("indexing", "该来源内容已存在，跳过重复入库。", 100, source_name, expected_chunk_count, len(cleaned_text))
                return {
                    "status": "skipped",
                    "message": f"来源 '{source_name}' 内容未变化，已跳过重复处理",
                    "source_name": source_name,
                    "chunk_count": expected_chunk_count,
                    "total_chars": len(cleaned_text),
                    "templates_by_name": template_summary,
                    "progress": self.progress.copy(),
                }

        self._set_progress("cleaning", "正在清洗 Wiki/HTML 标记并去除噪声...", 30, source_name, total_chars=len(cleaned_text))

        chunks = chunk_text(cleaned_text)
        self._set_progress("chunking", "文本已切分为多个知识块，准备生成 embedding...", 60, source_name, chunk_count=len(chunks), total_chars=len(cleaned_text))

        if not chunks:
            raise ValueError("清洗后没有可入库的文本内容")

        pending = [(index, chunk) for index, chunk in enumerate(chunks) if index not in existing]
        for batch_start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
            batch = pending[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
            batch_embeddings = self._batch_embedding([chunk for _, chunk in batch])
            ids = [f"{source_name}-{source_hash[:16]}-{index}" for index, _ in batch]
            metadatas = [
                {
                    "source": source_name,
                    "source_hash": source_hash,
                    "chunk_index": index,
                    "length": len(chunk),
                    "template_names": ",".join(sorted({template.name for template in parsed_document.templates})),
                }
                for index, chunk in batch
            ]
            self._set_progress("embedding", f"正在生成 embedding（{min(batch_start + len(batch), len(pending))}/{len(pending)}）...", 60 + int((min(batch_start + len(batch), len(pending)) / len(chunks)) * 30), source_name, len(chunks), len(cleaned_text))
            self._set_progress("indexing", f"正在写入 chunk（{min(batch_start + len(batch), len(pending))}/{len(chunks)}）...", 95, source_name, len(chunks), len(cleaned_text))
            self.collection.add(documents=[chunk for _, chunk in batch], embeddings=batch_embeddings, metadatas=metadatas, ids=ids)

        # 新版本全部成功后再清理旧版本，失败时保留原有可检索数据。
        self._delete_old_source(source_name, source_hash)

        processed_path = self._save_processed_text(f"{source_name}.txt", cleaned_text)
        self._set_progress("indexing", "清洗后的文本和 embedding 已写入 ChromaDB，流程完成。", 100, source_name, chunk_count=len(chunks), total_chars=len(cleaned_text))

        return {
            "status": "success",
            "source_name": source_name,
            "processed_path": processed_path,
            "chunk_count": len(chunks),
            "total_chars": len(cleaned_text),
            "heading_count": len(parsed_document.headings),
            "link_count": len(parsed_document.links),
            "template_count": len(parsed_document.templates),
            "templates_by_name": template_summary,
            "progress": self.progress.copy(),
        }

    def prepare_from_raw_dir(self, raw_dir: str = "data/raw") -> Dict[str, Any]:
        """批量处理目录下所有 `.txt` 文件。

        输入：`raw_dir`，原始文本目录路径。
        输出：包含成功状态、处理文件数量和每个文件结果的汇总字典。
        异常：目录不存在或任一文件处理失败时抛出异常。
        """
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
    """创建数据准备管线并批量处理原始文本目录。

    输入：`raw_dir`，原始文本目录路径。
    输出：`DataPreparationPipeline.prepare_from_raw_dir` 返回的汇总字典。
    """
    pipeline = DataPreparationPipeline()
    return pipeline.prepare_from_raw_dir(raw_dir)
