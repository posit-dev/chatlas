import os

from mcp.server.mcpserver import MCPServer

app = MCPServer("test")


@app.tool(description="Add two numbers.")
def add(x: int, y: int) -> int:
    return x + y


app.run(transport="streamable-http", port=int(os.getenv("MCP_PORT", "8000")))
