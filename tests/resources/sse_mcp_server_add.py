from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test")

@mcp.tool(description="Add two numbers.")
def add(x: int, y: int) -> int:
    return x + y

if __name__ == "__main__":
    mcp.run(transport='sse')