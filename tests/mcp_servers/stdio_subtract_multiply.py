from mcp.server.mcpserver import MCPServer

app = MCPServer("test")


@app.tool(description="Subtract two numbers.")
def subtract(y: int, z: int) -> int:
    return y - z


@app.tool(description="Multiply two numbers.")
def multiply(a: int, b: int) -> int:
    return a * b


app.run(transport="stdio")
