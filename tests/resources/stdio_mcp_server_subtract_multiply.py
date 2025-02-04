from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test")

@mcp.tool(description="Subtract two numbers.")
def subtract(y: int, z: int) -> int:
    return y - z

@mcp.tool(description="Multiply two numbers.")
def multiply(a: int, b: int) -> int:
    return a * b

if __name__ == "__main__":
    mcp.run(transport='stdio')