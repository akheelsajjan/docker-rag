from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import asyncio
from fastmcp import Client

from app.graph import app_graph
from app.guardrails import redact_pii

app = FastAPI(title="RAG LangGraph Assistant")


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    thread_id: str = "default-thread"


class ChatResponse(BaseModel):
    answer: str
    classification: str | None = None
    guardrail_status: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    clean_input, _ = redact_pii(req.message)
    config = {"configurable": {"thread_id": req.thread_id}}

    result = app_graph.invoke(
        {"messages": [HumanMessage(content=clean_input)], "user_id": req.user_id},
        config,
    )

    snapshot = app_graph.get_state(config)
    guardrail_status = snapshot.values.get("guardrail_status")
    classification = snapshot.values.get("classification")

    return ChatResponse(
        answer=result["messages"][-1].content,
        classification=classification,
        guardrail_status=guardrail_status,
    )

async def call_mcp_add(a: int, b: int):
    async with Client("http://mcp-server:8001/mcp") as client:
        return await client.call_tool(
            "add",
            {"a": a, "b": b}
        )


def mcp_add(a: int, b: int) -> int:
    result = asyncio.run(call_mcp_add(a, b))
    return result.data[0].text