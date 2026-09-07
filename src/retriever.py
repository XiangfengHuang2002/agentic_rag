import requests
import chromadb
import numpy as np
from typing import List, Dict, Any
from src.config import (
    SILICONFLOW_API_KEY,
    SILICONFLOW_BASE_URL,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    CHROMA_DB_DIR
)

class WikiRetriever:
    def __init__(self, collection_name: str = "game_wiki"):
        """初始化 ChromaDB 集合和 SiliconFlow 请求头。

        输入：`collection_name`，要读取或创建的 ChromaDB 集合名。
        输出：无；集合客户端和 HTTP 请求配置保存到实例属性。
        """
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
            "Content-Type": "application/json"
        }

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度。

        输入：`vec1`、`vec2`，长度相同的数值向量列表。
        输出：浮点型余弦相似度。
        """
        v1, v2 = np.array(vec1), np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

    def _get_query_embedding(self, query: str) -> List[float]:
        """调用 Embedding API 将用户问题转换为向量。

        输入：`query`，待向量化的自然语言问题。
        输出：模型返回的浮点向量列表。
        异常：HTTP 请求失败或响应格式不正确时抛出异常。
        """
        url = f"{SILICONFLOW_BASE_URL}/embeddings"
        payload = {
            "model": EMBEDDING_MODEL,
            "input": query,
            "encoding_format": "float"
        }
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def _rerank_documents(self, query: str, documents: List[str]) -> List[Dict[str, Any]]:
        """调用 Reranker API 对候选文档进行交叉重排。

        输入：`query`，用户问题；`documents`，候选文档文本列表。
        输出：按相关性排序的结果字典列表，包含文档文本和重排分数。
        异常：HTTP 请求失败或响应格式不正确时抛出异常。
        """
        url = f"{SILICONFLOW_BASE_URL}/rerank"
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": True
        }
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()["results"]

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """召回、重排并返回带相似度分数的知识片段。

        输入：`query`，用户问题；`top_k`，ChromaDB 初筛候选数量。
        输出：结果字典列表，每项包含内容、元数据、`rerank_score` 和 `vector_sim`；外部 API 失败时返回空列表。
        """
        print(f"收到用户提问: '{query}'，开始进行实时向量化并检索...")
        
        try:
            query_embedding = self._get_query_embedding(query)
        except Exception as e:
            print(f"在线计算提问向量失败: {e}")
            return []

        # 增加 include=["embeddings"] 以便手动计算相似度
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings"]
        )
        
        if not results or not results["documents"] or not results["documents"][0]:
            print("向量库无任何匹配候选。")
            return []
            
        raw_docs = results["documents"][0]
        raw_metadatas = results["metadatas"][0]
        raw_embeddings = results["embeddings"][0]
        
        print(f"粗筛召回 {len(raw_docs)} 条候选片段，开始调用 Reranker 进行交叉重排...")
        
        try:
            rerank_results = self._rerank_documents(query, raw_docs)
        except Exception as e:
            print(f"调用重排模型失败: {e}")
            return []

        formatted_results = []
        # 构建文档映射以获取向量
        doc_emb_map = {doc: emb for doc, emb in zip(raw_docs, raw_embeddings)}
        doc_meta_map = {doc: meta for doc, meta in zip(raw_docs, raw_metadatas)}
        
        for item in rerank_results:
            doc_text = item["document"]["text"]
            rerank_score = item["relevance_score"]
            
            # 手动计算余弦相似度，确保与 v3 完全一致
            sim = self._cosine_similarity(query_embedding, doc_emb_map[doc_text])
            
            print(f"   [重排结果] 相似度: {sim:.4f}, 重排得分: {rerank_score:.4f}")
            formatted_results.append({
                "content": doc_text,
                "metadata": doc_meta_map.get(doc_text, {}),
                "rerank_score": rerank_score,
                "vector_sim": sim  # 此处传出的是纯净的 0-1 相似度
            })
            
        print(f"成功输出 {len(formatted_results)} 条已排序的带分数片段。\n")
        return formatted_results