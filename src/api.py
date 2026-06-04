import json
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from src.agent import GameAgent

agent = None
THRESHOLD = 0.5  # 对齐 v3 的判定阈值

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    print("正在初始化智能体并加载向量数据库客户端...")
    try:
        agent = GameAgent()
        print("智能体 API 服务成功拉起，准备就绪。")
    except Exception as e:
        print(f"智能体初始化失败: {e}")
        raise e
    yield
    print("智能体 API 服务正在关闭...")

app = FastAPI(title="Game Wiki RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def chat_stream_generator(query: str):
    print(f"API 接收到流式请求，用户提问: {query}")
    
    try:
        yield {"event": "status", "data": json.dumps({"message": "正在进行实时向量化并检索..."})}
        await asyncio.sleep(0.1)
        
        retrieved_chunks = agent.retriever.search(query, top_k=5)
        
        if not retrieved_chunks:
            yield {"event": "decision", "data": json.dumps({"need_rag": False, "score": 0.0, "reason": "未检索到任何背景片段"})}
            chunks_to_llm = []
        else:
            highest_vector_sim = float(max(c["vector_sim"] for c in retrieved_chunks))
            need_rag = bool(highest_vector_sim >= THRESHOLD)
            
            chunks_data = [
                {
                    "content": c["content"],
                    "vector_sim": float(c["vector_sim"]),
                    "rerank_score": float(c["rerank_score"])
                } for c in retrieved_chunks
            ]
            
            yield {
                "event": "decision", 
                "data": json.dumps({
                    "need_rag": need_rag, 
                    "score": float(highest_vector_sim),
                    "threshold": THRESHOLD,
                    "chunks": chunks_data
                })
            }
            chunks_to_llm = retrieved_chunks if need_rag else []
            await asyncio.sleep(0.1)

        yield {"event": "status", "data": json.dumps({"message": "正在调用大模型生成最终回复..."})}
        
        context_text = "\n---\n".join([chunk["content"] for chunk in chunks_to_llm]) if chunks_to_llm else ""
        if context_text:
            system_prompt = f"你是一个精通游戏机制的助手。请严格根据以下提供的背景知识回答玩家的问题。\n背景知识：\n{context_text}"
        else:
            system_prompt = "你是一个精通游戏机制的助手。如果玩家提问的内容不是游戏相关的机制或你无法确定的内容，请直接回答不知道。"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        answer = agent._call_llm(messages)
        
        yield {"event": "result", "data": json.dumps({"answer": answer})}
        
    except Exception as e:
        print(f"流式 API 内部发生异常: {e}")
        yield {"event": "error", "data": json.dumps({"detail": str(e)})}

@app.get("/api/chat/stream")
def chat_stream(query: str = Query(..., description="用户提问内容")):
    if not query.strip():
        raise HTTPException(status_code=400, detail="提问内容不能为空")
    return EventSourceResponse(chat_stream_generator(query))

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)