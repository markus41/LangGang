"""
Script to start the LangGang MCP server.

This script can be used to test the MCP server locally or start it for use with MCP clients.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Main entry point for starting MCP servers."""
    parser = argparse.ArgumentParser(description="Start LangGang MCP servers")
    parser.add_argument(
        "server",
        choices=["templates", "llm-cli", "docs"],
        help="Which MCP server to start"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (print info and exit)"
    )
    
    args = parser.parse_args()
    
    if args.server == "templates":
        if args.test:
            print("Template MCP Server")
            print("Command: python -m langgang.mcp_server_runner")
            print("Tools: list_templates, generate_from_cookiecutter, generate_from_copier, generate_from_maven_archetype")
        else:
            from langgang import mcp_server_runner
            import asyncio
            asyncio.run(mcp_server_runner.main())
    
    elif args.server == "llm-cli":
        if args.test:
            print("LLM CLI MCP Server")
            print("Command: python -m langgang.llm_cli_mcp_server_runner")
            print("Tools: list_available_clis, execute_gemini_cli, execute_codex_cli, execute_openai_cli, gemini_chat, codex_generate, openai_chat")
        else:
            from langgang import llm_cli_mcp_server_runner
            import asyncio
            asyncio.run(llm_cli_mcp_server_runner.main())
    
    elif args.server == "docs":
        if args.test:
            print("Documentation MCP Server")
            print("Command: python -m langgang.docs_mcp_server_runner")
            print("Tools: search_langchain_docs, search_langgraph_docs, get_langchain_examples, get_langgraph_examples, list_available_docs")
        else:
            from langgang import docs_mcp_server_runner
            import asyncio
            asyncio.run(docs_mcp_server_runner.main())


if __name__ == "__main__":
    main()
