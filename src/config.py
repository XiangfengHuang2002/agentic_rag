import os
from dotenv import load_dotenv

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

LLM_MODEL = "Qwen/Qwen3-8B"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

VECTOR_SEARCH_THRESHOLD = float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.5"))
REACT_MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "3"))

CHROMA_DB_DIR = "./chroma_db"

# 每次发送到 embedding 服务的文本块数量。过大可能触发供应商请求限制。
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
# 单个 chunk 的字符上限，需小于模型/供应商允许的输入上限。
MAX_EMBEDDING_CHARS = int(os.getenv("MAX_EMBEDDING_CHARS", "1200"))
# 网络临时失败时的最大重试次数。
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))