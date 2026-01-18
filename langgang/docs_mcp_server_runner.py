"""
Documentation MCP Server Runner

This module implements a proper MCP server for LangChain and LangGraph documentation
access using the MCP SDK.
"""

import asyncio
import json
import logging
from typing import Any, Sequence

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP SDK not available. Install with: pip install mcp")

from langgang.docs_mcp_server import DocumentationMCPServer

logger = logging.getLogger(__name__)

# Initialize the documentation server
docs_server = DocumentationMCPServer()

# Create MCP server instance
if MCP_AVAILABLE:
    mcp_server = Server("langgang-docs")


    @mcp_server.list_tools()
    async def list_tools() -> list[Tool]:
        """List all available documentation tools."""
        return [
            Tool(
                name="search_langchain_docs",
                description="Search LangChain documentation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for LangChain documentation"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="search_langgraph_docs",
                description="Search LangGraph documentation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for LangGraph documentation"
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="get_langchain_examples",
                description="Get LangChain code examples by category",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Example category (chains, agents, tools, memory, or 'all')",
                            "default": "all"
                        }
                    }
                }
            ),
            Tool(
                name="get_langgraph_examples",
                description="Get LangGraph code examples by category",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Example category (graphs, nodes, state, persistence, or 'all')",
                            "default": "all"
                        }
                    }
                }
            ),
            Tool(
                name="list_available_docs",
                description="List all available documentation sources",
                inputSchema={
                    "type": "object",
                    "properties": {},
                }
            ),
        ]


    @mcp_server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
        """Handle tool calls."""
        try:
            if name == "search_langchain_docs":
                query = arguments.get("query", "")
                if not query:
                    return [TextContent(type="text", text="Error: query parameter is required")]
                result = docs_server.search_langchain_docs(query)
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            
            elif name == "search_langgraph_docs":
                query = arguments.get("query", "")
                if not query:
                    return [TextContent(type="text", text="Error: query parameter is required")]
                result = docs_server.search_langgraph_docs(query)
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            
            elif name == "get_langchain_examples":
                category = arguments.get("category", "all")
                result = docs_server.get_langchain_examples(category)
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            
            elif name == "get_langgraph_examples":
                category = arguments.get("category", "all")
                result = docs_server.get_langgraph_examples(category)
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            
            elif name == "list_available_docs":
                result = docs_server.list_available_docs()
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
        
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Main entry point for documentation MCP server."""
    if not MCP_AVAILABLE:
        logger.error("MCP SDK not available. Install with: pip install mcp")
        return
    
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting LangGang Documentation MCP Server...")
    logger.info(f"Available documentation sources: {list(docs_server.list_available_docs().keys())}")
    
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
            raise_exceptions=False
        )


if __name__ == "__main__":
    asyncio.run(main())
