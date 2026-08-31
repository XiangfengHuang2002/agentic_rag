import os
from dotenv import load_dotenv

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

LLM_MODEL = "Qwen/Qwen3-8B"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

VECTOR_SEARCH_THRESHOLD = float(os.getenv("VECTOR_SEARCH_THRESHOLD", "0.5"))
AGENT_MODE = os.getenv("AGENT_MODE", "rag").lower()
REACT_MAX_STEPS = int(os.getenv("REACT_MAX_STEPS", "3"))

CHROMA_DB_DIR = "./chroma_db"