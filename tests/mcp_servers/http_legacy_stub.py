"""Legacy-era (handshake-only) MCP stub server.

Speaks just enough of the 2025-11-25 protocol to exercise chatlas's legacy
fallback paths: answers `initialize`, `tools/list`, and `tools/call` with
plain JSON responses, and rejects `server/discover` with "method not found"
(which is what a pre-2026 server does).
"""

import asyncio
import json
import os

from uvicorn import Config, Server


async def app(scope, receive, send):
    assert scope["type"] == "http"
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    request = json.loads(body or b"{}")

    async def respond(payload, status=200):
        data = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": data})

    method = request.get("method")
    if method == "initialize":
        await respond(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "legacy_stub", "version": "0.0.0"},
                },
            }
        )
    elif method == "notifications/initialized":
        await respond({}, status=202)
    elif method == "tools/list":
        await respond(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo back the input.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        }
                    ]
                },
            }
        )
    elif method == "tools/call":
        text = request["params"]["arguments"]["text"]
        await respond(
            {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                },
            }
        )
    else:
        # server/discover and anything else: pre-2026 servers don't know it
        await respond(
            {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32601, "message": "Method not found"},
            }
        )


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8000"))
    asyncio.run(
        Server(Config(app, host="127.0.0.1", port=port, log_level="warning")).serve()
    )
