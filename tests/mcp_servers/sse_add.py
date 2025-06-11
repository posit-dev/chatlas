import os

from mcp.server.fastmcp import FastMCP

app = FastMCP("test", port=int(os.getenv("MCP_PORT", "8000")))


@app.tool(description="Add two numbers.")
def add(x: int, y: int) -> int:
    return x + y


app.run(transport="streamable-http")
