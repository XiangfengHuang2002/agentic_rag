import os
from dotenv import load_dotenv

load_dotenv()

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"

LLM_MODEL = "tencent/Hunyuan-MT-7B"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
VECTOR_SEARCH_THRESHOLD = 0.5

CHROMA_DB_DIR = "./chroma_db"