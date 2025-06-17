import os

from mcp.server.fastmcp import FastMCP

app = FastMCP("test", port=int(os.getenv("MCP_PORT", "8000")))


@app.tool(description="Gets the current date")
async def get_current_date():
    import asyncio

    await asyncio.sleep(0.1)
    return "2024-01-01"


app.run(transport="streamable-http")
