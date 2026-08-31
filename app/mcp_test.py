import asyncio
from app.tools import mcp_add_tool


async def main():
    result = await mcp_add_tool.ainvoke({
        "a": 10,
        "b": 20
    })

    print("RESULT:", result)


asyncio.run(main())