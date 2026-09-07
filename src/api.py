import json
import asyncio
import uvicorn
from uuid import uuid4
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from src.langgraph_orchestrator import LangGraphAgent as OrchestratorAgent
from src.data_preparation import DataPreparationPipeline

agent = None


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
        agent = OrchestratorAgent()
        print("使用 LangGraph ReAct 编排器启动智能体")

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

        if not hasattr(current_agent, "run_react_query"):
            raise RuntimeError("ReAct 编排器未正确初始化")

        yield {"event": "decision", "data": json.dumps({
            "mode": "react",
            "reason": "ReAct 正在决定是否调用 RAG 检索节点",
        }, ensure_ascii=False)}
        yield {"event": "node", "data": json.dumps({
            "node": "agent",
            "message": "ReAct 正在决定下一步动作",
            "step": 1,
            "action": "decide",
        }, ensure_ascii=False)}
        answer = current_agent.run_react_query(query)
        yield {"event": "result", "data": json.dumps({"answer": answer, "mode": "react"}, ensure_ascii=False)}
        return
    except Exception as e:
        print(f"流式 API 内部发生异常: {e}")
        yield {"event": "error", "data": json.dumps({"detail": str(e)})}


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