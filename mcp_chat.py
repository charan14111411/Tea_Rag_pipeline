"""
A small AI assistant that uses the tea MCP server - for testing MCP locally.

The assistant's model (Groq) knows nothing about tea books by itself. It connects
to the MCP server, reads the tool list (ask_tea_expert, search_tea_books,
list_tea_books) and decides on its own when to call them. Every tool call is
printed, so you can see the MCP traffic.

  python mcp_chat.py                               # starts mcp_server.py itself (stdio)
  python mcp_chat.py --url http://localhost:8000/mcp   # uses the running web app's /mcp endpoint
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

SYSTEM = (
    "You are a helpful farm assistant. You have tools from a tea book library (via MCP). "
    "For any question about tea cultivation, pests, diseases, soil, clones, plucking or processing, "
    "use the tools instead of your own knowledge, and keep the book sources in your reply. "
    "For follow-up questions to ask_tea_expert, pass the conversation_id it returned earlier."
)
MAX_TOOL_ROUNDS = 5


def to_groq_tool(tool):
    """MCP tool description -> OpenAI/Groq function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        },
    }


def tool_result_text(result):
    if result.structured_content is not None:
        return json.dumps(result.structured_content, ensure_ascii=False)
    return "\n".join(getattr(c, "text", str(c)) for c in result.content)


async def chat(server):
    async with Client(server, read_timeout_seconds=300) as client:
        tools = (await client.list_tools()).tools
        print("Connected to MCP server. Tools:", ", ".join(t.name for t in tools))
        print("Type 'exit' to quit.\n")

        model = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0
        ).bind_tools([to_groq_tool(t) for t in tools])
        messages = [SystemMessage(SYSTEM)]

        while True:
            try:
                question = (await asyncio.to_thread(input, "You: ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question.lower() == "exit":
                break
            messages.append(HumanMessage(question))

            for _ in range(MAX_TOOL_ROUNDS):
                reply = await model.ainvoke(messages)
                messages.append(reply)
                if not reply.tool_calls:
                    break
                for call in reply.tool_calls:
                    print(f"  [MCP] {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
                    result = await client.call_tool(call["name"], call["args"])
                    text = tool_result_text(result)
                    status = "error" if result.is_error else f"{len(text)} chars"
                    print(f"  [MCP] <- {status}")
                    messages.append(ToolMessage(text, tool_call_id=call["id"]))

            print(f"\nAssistant: {reply.content}\n")


def main():
    parser = argparse.ArgumentParser(description="Test chat assistant for the tea MCP server")
    parser.add_argument("--url", help="MCP endpoint URL, e.g. http://localhost:8000/mcp (default: start mcp_server.py)")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY is not set (put it in .env).")

    if args.url:
        asyncio.run(chat(args.url))
        return

    # Start mcp_server.py as a subprocess. Its logs go to mcp_server.log instead
    # of the terminal, so the chat stays readable.
    params = StdioServerParameters(command=sys.executable, args=[str(PROJECT_ROOT / "mcp_server.py")])
    with open(PROJECT_ROOT / "mcp_server.log", "w", encoding="utf-8") as errlog:
        print("Starting MCP server (logs: mcp_server.log)...")
        asyncio.run(chat(stdio_client(params, errlog=errlog)))


if __name__ == "__main__":
    main()
