from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test")

@mcp.tool(description="Gets the current date")
async def get_current_date():
    import asyncio

    await asyncio.sleep(0.1)
    return "2024-01-01"

if __name__ == "__main__":
    mcp.run(transport='sse')