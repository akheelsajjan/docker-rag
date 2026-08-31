from langchain.tools import tool

from app.rag import vectorstore, llm
from fastmcp import Client


async def mcp_add_tool(a: float, b: float) -> float:
    async with Client("http://mcp-server:8001/mcp") as client:
        result = await client.call_tool(
            "add",
            {"a": a, "b": b}
        )
        print(">>> MCP RESULT:", result)
        return float(result.data)


mcp_add_tool = tool(
    mcp_add_tool,
    description="Add two numbers using the MCP server."
)


@tool
def search_local_docs(query: str) -> str:
    """
    Search the internal knowledge base (Apple Inc. and Apple fruit documents).
    Use only once per question. Do not call repeatedly.
    """
    docs = vectorstore.similarity_search(query, k=3)

    if not docs:
        return "No relevant documents found."

    response = "\n\n".join(
        f"[Chunk {i+1}]\n{doc.page_content[:700]}"
        for i, doc in enumerate(docs)
    )
    return response[:2500]


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        return float("nan")
    return a / b


tools = [mcp_add_tool, subtract, multiply, divide, search_local_docs]
llm_with_tools = llm.bind_tools(tools, tool_choice="auto")