import os

from mcp.server.fastmcp import FastMCP

app = FastMCP("test", port=int(os.getenv("MCP_PORT", "8000")))


@app.tool(description="Subtract two numbers.")
def subtract(y: int, z: int) -> int:
    return y - z


@app.tool(description="Multiply two numbers.")
def multiply(a: int, b: int) -> int:
    return a * b


app.run(transport="stdio")
