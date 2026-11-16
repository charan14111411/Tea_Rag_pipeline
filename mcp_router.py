"""
Run several MCP servers behind ONE ngrok tunnel (one free ngrok domain).

ngrok can forward its domain to only one local port. This router listens on that
port and sends each request to the right MCP server by its secret path:

    ngrok http 8080  ->  mcp_router.py (port 8080)
                           /mcp-<tea secret>     -> tea MCP server    (127.0.0.1:8002)
                           /mcp-<coffee secret>  -> coffee MCP server (127.0.0.1:8001)
                           anything else         -> 404

Each server's secret path is read from that project's own .env (MCP_PATH), so the
secrets are not copied anywhere; after changing one, just restart the router.
Requests are passed through unchanged (including the Host header, which each MCP
server checks) and responses are streamed, so MCP's server-sent events work.

    python mcp_router.py                 # routes from mcp_router.json, listens on 8080
"""
import json
import logging
import sys
from pathlib import Path

import httpx
import uvicorn
from dotenv import dotenv_values
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import PlainTextResponse, StreamingResponse
from starlette.routing import Route

CONFIG = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "mcp_router.json"
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailer", "transfer-encoding", "upgrade", "content-length"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp_router")


def load_routes(config):
    """{secret path: (name, target base URL)} from each project's .env MCP_PATH."""
    routes = {}
    for r in config["routes"]:
        env_file = Path(r["env_file"])
        if not env_file.exists():
            sys.exit(f"[{r['name']}] .env not found: {env_file}")
        path = (dotenv_values(env_file).get("MCP_PATH") or "").strip()
        if not path.startswith("/") or len(path) < 20:
            sys.exit(f"[{r['name']}] MCP_PATH missing or too short in {env_file}")
        if path in routes:
            sys.exit(f"[{r['name']}] has the same MCP_PATH as [{routes[path][0]}] - each needs its own")
        routes[path] = (r["name"], r["target"].rstrip("/"))
    return routes


config = json.loads(CONFIG.read_text(encoding="utf-8"))
ROUTES = load_routes(config)
# No read timeout: MCP can keep a server-sent-events stream open for a long time.
client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=None, write=60, pool=None))


async def proxy(request: Request):
    route = ROUTES.get(request.url.path)
    if route is None:
        return PlainTextResponse("Not Found", status_code=404)
    name, target = route

    url = target + request.url.path + (f"?{request.url.query}" if request.url.query else "")
    # Keep the original Host header (the tunnel's hostname): each MCP server checks it.
    headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP]
    try:
        upstream = await client.send(
            client.build_request(request.method, url, headers=headers, content=request.stream()),
            stream=True,
        )
    except httpx.ConnectError:
        logger.warning(f"{name}: server not running at {target}")
        return PlainTextResponse(f"{name} MCP server is not running", status_code=502)

    logger.info(f"{name}: {request.method} -> {upstream.status_code}")
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers={k: v for k, v in upstream.headers.items() if k.lower() not in HOP_BY_HOP},
        background=BackgroundTask(upstream.aclose),
    )


app = Starlette(routes=[Route("/{path:path}", proxy, methods=["GET", "POST", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"])])


if __name__ == "__main__":
    port = config.get("listen_port", 8080)
    for path, (name, target) in ROUTES.items():
        logger.info(f"{name}: {path[:5]}<secret> -> {target}")
    logger.info(f"Router on http://127.0.0.1:{port} - run: ngrok http {port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
