
"""Larkspur Transit MCP server."""

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage

from transit_kit.sql_tools import (
    list_tables as db_list_tables,
    get_table_schema as db_get_table_schema,
    execute_sql as db_execute_sql,
)

mcp = FastMCP("Larkspur-Transit-Ops")


# Task 1: MCP tools

@mcp.tool()
def list_tables() -> list[str]:
    """List available database tables."""
    return db_list_tables()


@mcp.tool()
def get_table_schema(table_name: str) -> list[dict]:
    """Get the columns and types of an allowed table."""
    return db_get_table_schema(table_name)


@mcp.tool()
def execute_sql(query: str) -> str:
    """Execute a read-only SQL query and return formatted results."""
    return db_execute_sql(query)


# Task 2: Database overview resource

@mcp.resource("transit://overview", mime_type="text/plain")
def transit_overview() -> str:
    """Return current row counts for all available tables."""
    lines = ["Larkspur Transit operations database"]

    for table in db_list_tables():
        safe_table = table.replace('"', '""')
        result = db_execute_sql(
            f'SELECT COUNT(*) AS row_count FROM "{safe_table}"'
        )

        if result.startswith("SQL Error:"):
            raise RuntimeError(
                f"Unable to count rows for {table}: {result}"
            )

        rows = result.splitlines()
        if len(rows) < 2:
            raise RuntimeError(
                f"Unexpected row-count result for {table}: {result}"
            )

        count = rows[-1].split("|")[-1].strip()
        lines.append(f"{table}: {count} rows")

    return "\n".join(lines)


# Task 3: Analyst prompt

@mcp.prompt(
    name="transit_analyst",
    description="Guide safe transit database analysis.",
)
def transit_analyst(question: str) -> list[UserMessage]:
    return [
        UserMessage(
            content=(
                "You are the Larkspur Transit operations analyst.\n\n"
                "1. Call list_tables to discover available tables.\n"
                "2. Call get_table_schema for relevant tables.\n"
                "3. Write a read-only SQLite SELECT query.\n"
                "4. Execute it using execute_sql.\n"
                "5. Answer using actual results without inventing data.\n\n"
                f"QUESTION: {question}"
            )
        )
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
