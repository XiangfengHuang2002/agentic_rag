import json
import asyncio
import uvicorn
from uuid import uuid4
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from src.config import AGENT_MODE, VECTOR_SEARCH_THRESHOLD
from src.langgraph_orchestrator import LangGraphAgent as OrchestratorAgent
from src.agent import GameAgent
from src.data_preparation import DataPreparationPipeline

agent = None
THRESHOLD = float(VECTOR_SEARCH_THRESHOLD)


async def _run_preparation_task(pipeline: DataPreparationPipeline, raw_text: str, source_name: str):
    try:
        await asyncio.to_thread(pipeline.prepare_from_text, raw_text, source_name)
    except Exception as exc:
        pipeline.progress.update({
            "status": "failed",
            "stage": "error",
            "message": str(exc),
            "percent": 0,
            "source_name": source_name,
        })


async def _run_raw_import_task(pipeline: DataPreparationPipeline):
    try:
        await asyncio.to_thread(pipeline.prepare_from_raw_dir, "data/raw")
    except Exception as exc:
        pipeline.progress.update({
            "status": "failed",
            "stage": "error",
            "message": str(exc),
            "percent": 0,
            "source_name": "data/raw",
        })

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    print("正在初始化智能体并加载向量数据库客户端...")
    try:
        if OrchestratorAgent is not None:
            agent = OrchestratorAgent()
            print("使用 LangGraph 编排器启动智能体")
        else:
            agent = GameAgent()
            print("使用常规模型 GameAgent 启动智能体（未检测到 LangGraph 编排器）")

        app.state.agent = agent
        app.state.data_pipeline = DataPreparationPipeline()
        print("智能体 API 服务成功拉起，准备就绪。")
    except Exception as e:
        print(f"智能体初始化失败: {e}")
        raise e
    yield
    print("智能体 API 服务正在关闭...")

app = FastAPI(title="游戏知识 RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

async def chat_stream_generator(query: str):
    print(f"API 接收到流式请求，用户提问: {query}")
    current_agent = app.state.agent if hasattr(app.state, "agent") else agent
    
    try:
        yield {"event": "status", "data": json.dumps({"message": "正在进行实时向量化并检索..."})}
        await asyncio.sleep(0.1)

        if current_agent is None:
            raise RuntimeError("智能体尚未初始化完成，请检查应用启动状态")

        if AGENT_MODE != "rag" and hasattr(current_agent, "run_react_query"):
            retrieved_chunks = current_agent.retriever.search(query, top_k=5)
            decision_data = _build_decision_data(retrieved_chunks)
            decision_data["mode"] = "react"
            decision_data["need_rag"] = bool(retrieved_chunks)
            decision_data["reason"] = (
                "ReAct 决策中：命中检索证据，选择 RAG 分支继续回答"
                if retrieved_chunks
                else "ReAct 决策中：未命中证据，退回到一般性回答或继续推理"
            )

            yield {"event": "node", "data": json.dumps({
                "node": "retrieve",
                "message": f"初始检索召回 {len(retrieved_chunks)} 条候选",
            }, ensure_ascii=False)}
            yield {"event": "decision", "data": json.dumps(decision_data, ensure_ascii=False)}
            yield {"event": "node", "data": json.dumps({
                "node": "react",
                "message": "正在执行 ReAct 推理循环：先观察召回证据，再决定下一步动作",
                "step": 1,
                "action": "search",
                "score": decision_data.get("score", 0.0),
                "need_rag": decision_data["need_rag"],
                "evidence_count": len(retrieved_chunks),
            }, ensure_ascii=False)}
            answer = current_agent.run_react_query(query, retrieved_chunks)
            yield {"event": "result", "data": json.dumps({"answer": answer, "mode": "react"})}
            return
        
        retrieved_chunks = current_agent.retriever.search(query, top_k=5)
        # 发送检索节点事件
        try:
            yield {"event": "node", "data": json.dumps({"node": "retrieve", "message": f"粗筛召回 {len(retrieved_chunks)} 条候选"})}
        except Exception:
            pass
        
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
            
            # 发送重排/判定节点事件摘要
            try:
                yield {"event": "node", "data": json.dumps({"node": "decide", "message": f"最高相似度 {highest_vector_sim:.4f}，阈值 {THRESHOLD}"})}
            except Exception:
                pass

            yield {
                "event": "decision",
                "data": json.dumps({
                    "need_rag": need_rag,
                    "score": float(highest_vector_sim),
                    "threshold": THRESHOLD,
                    "reason": "高于阈值，转入知识增强回答" if need_rag else "低于阈值，转通用回答",
                    "chunks": chunks_data
                })
            }
            chunks_to_llm = retrieved_chunks if need_rag else []
            await asyncio.sleep(0.1)

        # 发送模型调用节点事件
        try:
            yield {"event": "node", "data": json.dumps({"node": "call_llm", "message": "正在调用大模型生成最终回复"})}
        except Exception:
            pass
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
        
        # 支持原始的 GameAgent（提供 _call_llm）
        # 以及基于 LangGraph 的编排器（提供 run_query）。
        if hasattr(current_agent, "_call_llm"):
            answer = current_agent._call_llm(messages)
        elif hasattr(current_agent, "run_query"):
            # 若 agent 包装了 base_agent（如 LangGraphAgent），优先调用其 base_agent._call_llm
            # 并传入已构建的 `messages`，避免重复检索。
            if hasattr(current_agent, "base_agent") and hasattr(current_agent.base_agent, "_call_llm"):
                answer = current_agent.base_agent._call_llm(messages)
            else:
                # run_query 会在内部完成检索与判定，直接传入用户 query。
                answer = current_agent.run_query(query)
        else:
            raise RuntimeError("Agent 未暴露兼容的调用接口")
        
        yield {"event": "result", "data": json.dumps({"answer": answer})}
        
    except Exception as e:
        print(f"流式 API 内部发生异常: {e}")
        yield {"event": "error", "data": json.dumps({"detail": str(e)})}


def _build_decision_data(retrieved_chunks: list) -> dict:
    """整理前端需要的检索结果与门控信息。"""
    if not retrieved_chunks:
        return {
            "need_rag": False,
            "score": 0.0,
            "threshold": THRESHOLD,
            "chunks": [],
            "reason": "未检索到任何背景片段",
        }

    score = float(max(chunk.get("vector_sim", 0.0) for chunk in retrieved_chunks))
    need_rag = score >= THRESHOLD
    return {
        "need_rag": need_rag,
        "score": score,
        "threshold": THRESHOLD,
        "reason": "高于阈值，转入知识增强回答" if need_rag else "低于阈值，转通用回答",
        "chunks": [
            {
                "content": chunk.get("content", ""),
                "vector_sim": float(chunk.get("vector_sim", 0.0)),
                "rerank_score": float(chunk.get("rerank_score", 0.0)),
            }
            for chunk in retrieved_chunks
        ],
    }

@app.get("/api/chat/stream")
def chat_stream(query: str = Query(..., description="用户提问内容")):
    if not query.strip():
        raise HTTPException(status_code=400, detail="提问内容不能为空")
    return EventSourceResponse(chat_stream_generator(query))


@app.get("/api/data/prepare/progress")
async def get_data_preparation_progress():
    pipeline = getattr(app.state, "data_pipeline", None)
    if pipeline is None:
        pipeline = DataPreparationPipeline()
        app.state.data_pipeline = pipeline
    return pipeline.progress


@app.post("/api/data/prepare")
async def prepare_data_source(
    raw_text: str | None = Form(default=None),
    source_name: str = Form(default="uploaded_text"),
    file: UploadFile | None = File(default=None),
):
    pipeline = getattr(app.state, "data_pipeline", None)
    if pipeline is None:
        pipeline = DataPreparationPipeline()
        app.state.data_pipeline = pipeline

    try:
        if file is not None and file.filename:
            text = (await file.read()).decode("utf-8", errors="ignore")
            source_name = file.filename.rsplit(".", 1)[0] if "." in file.filename else file.filename
        elif raw_text is not None and raw_text.strip():
            text = raw_text
        else:
            raise ValueError("请提供粘贴文本或上传文本文件")

        job_id = str(uuid4())
        pipeline.progress.update({
            "job_id": job_id,
            "status": "queued",
            "stage": "received",
            "message": "任务已排队，准备开始数据处理...",
            "percent": 5,
            "source_name": source_name,
            "input_chars": len(text),
        })
        asyncio.create_task(_run_preparation_task(pipeline, text, source_name))
        return {"status": "accepted", "job_id": job_id, "progress": pipeline.progress.copy()}
    except Exception as exc:
        pipeline.progress.update({
            "status": "failed",
            "stage": "error",
            "message": str(exc),
            "percent": 0,
            "source_name": source_name,
        })
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/data/raw/import")
async def import_raw_directory():
    pipeline = getattr(app.state, "data_pipeline", None)
    if pipeline is None:
        pipeline = DataPreparationPipeline()
        app.state.data_pipeline = pipeline

    try:
        job_id = str(uuid4())
        pipeline.progress.update({
            "job_id": job_id,
            "status": "queued",
            "stage": "received",
            "message": "data/raw 批量导入任务已排队...",
            "percent": 5,
            "source_name": "data/raw",
        })
        asyncio.create_task(_run_raw_import_task(pipeline))
        return {"status": "accepted", "job_id": job_id, "progress": pipeline.progress.copy()}
    except Exception as exc:
        pipeline.progress.update({
            "status": "failed",
            "stage": "error",
            "message": str(exc),
            "percent": 0,
            "source_name": "data/raw",
        })
        raise HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)