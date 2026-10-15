"""Larkspur Transit MCP CLIENT and agent loop - your work goes here (Tasks 4 to 6).

This file is the client half of the lab. The server half is server.py, and the
two are separate programs: this one starts server.py as a subprocess and reaches
the database only by asking it. Do not import server.py from here.

What to do here:
  Task 4  TransitMCPClient      one server subprocess, one MCP session
  Task 5  mcp_tools_to_openai   MCP tool schemas to the OpenAI tools=[...] shape
  Task 6  run_agent_loop        one question to one answer, over the session

Add any helpers you like. The hard requirements are that everything sketched
below exists with these names and signatures, and behaves as the problem
statement describes.
"""
import asyncio
import json
import sys
import os
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl

from transit_kit import MODEL_NAME, make_client

SERVER_SCRIPT = "server.py"

# The most model calls run_agent_loop will ever make. One pass of the loop is one
# call to chat.completions.create, so a question needing k tool steps costs k + 1.
# Must be a whole number from 1 to 8; questions here need up to four tool steps,
# so anything below 5 will cut one short.
MAX_MODEL_TURNS = 8


class TransitMCPClient:
    """An MCP client bound to one stdio server process."""

    def __init__(self, command: Optional[str] = None,
                 args: Optional[List[str]] = None,
                 env: Optional[Dict[str, str]] = None) -> None:

        self.command = command or sys.executable
        self.args = args if args is not None else ["server.py"]
        self.env = env
        self.session = None
        self._stack = None

    async def connect(self) -> None:
        """Spawn server.py over stdio and initialise the MCP session."""
        if self.session is not None:
            return

        stack = AsyncExitStack()
        await stack.__aenter__()
        self._stack = stack

        try:
            server_env = os.environ.copy()
            if self.env is not None:
                server_env.update(self.env)

            params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=server_env,
            )

            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(params)
            )
            self.session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()

        except BaseException:
            self.session = None
            self._stack = None
            await stack.aclose()
            raise

    async def cleanup(self) -> None:
        """Close the session and the server process."""
        stack = self._stack
        self._stack = None
        self.session = None

        if stack is not None:
            await stack.aclose()

    async def __aenter__(self) -> "TransitMCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()
    
    def _require_session(self):
        if self.session is None:
            raise RuntimeError("MCP client is not connected")
        return self.session

    async def list_tools(self) -> List[Any]:
        """Return the tools the server advertises."""
        result = await self._require_session().list_tools()
        return result.tools

    async def call_tool(self, name: str,
                        arguments: Optional[Dict[str, Any]] = None) -> Any:
        """Run one tool on the server and return the result."""
        return await self._require_session().call_tool(name, arguments)

    async def read_resource(self, uri: str) -> Any:
        """Read one resource and return its content."""
        result = await self._require_session().read_resource(AnyUrl(uri))
        return "\n".join(
            item.text
            for item in result.contents
            if hasattr(item, "text")
        )

    async def list_prompts(self) -> List[Any]:
        """Return the prompts the server advertises."""
        result = await self._require_session().list_prompts()
        return result.prompts

    async def get_prompt(self, name: str,
                         arguments: Optional[Dict[str, str]] = None) -> Any:
        """Fetch one prompt with its arguments filled in, and return the messages."""
        result = await self._require_session().get_prompt(name, arguments)
        return result.messages


def mcp_tools_to_openai(mcp_tools) -> list[dict[str, Any]]:
    converted = []

    for tool in mcp_tools:
        schema = getattr(tool, "inputSchema", None)

        if not schema:
            schema = {"type": "object", "properties": {}}
        elif hasattr(schema, "model_dump"):
            schema = schema.model_dump()

        converted.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        })

    return converted

def _tool_result_text(result):
    blocks = getattr(result, "content", None) or []
    parts = [
        block.text
        for block in blocks
        if hasattr(block, "text")
    ]

    if parts:
        return "\n".join(parts)

    return str(result)


async def run_agent_loop(question: str) -> str:
    model_client = make_client()

    async with TransitMCPClient() as client:
        tools = mcp_tools_to_openai(await client.list_tools())

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the Larkspur Transit operations analyst. "
                    "First discover available tables and inspect the schemas "
                    "needed to answer the question. Use only read-only SELECT "
                    "queries through MCP tools. Base your answer on actual "
                    "query results and do not invent data."
                ),
            },
            {"role": "user", "content": question},
        ]

        for _ in range(MAX_MODEL_TURNS):
            response = model_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            if not message.tool_calls:
                return message.content or ""

            messages.append(message.model_dump(exclude_none=True))

            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    result = await client.call_tool(
                        tool_call.function.name,
                        arguments,
                    )

                    result_text = _tool_result_text(result)

                except Exception as exc:
                    result_text = f"Tool error: {exc}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

        return "Unable to produce a final answer within the model-turn limit."


if __name__ == "__main__":
    asked = sys.argv[1] if len(sys.argv) > 1 else "How many bus routes are there?"
    print(asyncio.run(run_agent_loop(asked)))
