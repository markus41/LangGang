# Starting the MCP Server

This guide explains how to start the LangGang MCP servers.

## Quick Start

### Option 1: Using the Helper Script

```bash
# Test server info
python scripts/start-mcp-server.py templates --test
python scripts/start-mcp-server.py llm-cli --test

# Start template server
python scripts/start-mcp-server.py templates

# Start LLM CLI server
python scripts/start-mcp-server.py llm-cli
```

### Option 2: Direct Python Module Execution

```bash
# Start template generation server
python -m langgang.mcp_server_runner

# Start LLM CLI server
python -m langgang.llm_cli_mcp_server_runner

# Start documentation server
python -m langgang.docs_mcp_server_runner
```

## MCP Client Integration

MCP servers are typically started automatically by MCP clients (like Cursor, Claude Desktop, etc.) when they connect. The servers communicate via stdio (standard input/output).

### Configuration

The servers are configured in `mcp-config.json`:

```json
{
  "mcpServers": {
    "langgang-templates": {
      "command": "python",
      "args": ["-m", "langgang.mcp_server_runner"]
    },
    "llm-cli": {
      "command": "python",
      "args": ["-m", "langgang.llm_cli_mcp_server_runner"]
    },
    "langchain-docs": {
      "command": "python",
      "args": ["-m", "langgang.docs_mcp_server_runner"]
    },
    "langgraph-docs": {
      "command": "python",
      "args": ["-m", "langgang.docs_mcp_server_runner"]
    }
  }
}
```

### Using with Cursor

1. Ensure `mcp-config.json` is in your project root
2. Cursor will automatically detect and start the MCP servers
3. The tools will be available in Cursor's AI chat interface

### Using with Claude Desktop

1. Add the server configuration to Claude Desktop's MCP settings
2. The servers will start automatically when Claude Desktop connects
3. Tools will be available in Claude's interface

## Available Tools

### Template Generation Server (`langgang-templates`)

- `list_templates` - List all available templates
- `generate_from_cookiecutter` - Generate from Cookiecutter template
- `generate_from_copier` - Generate from Copier template
- `generate_from_maven_archetype` - Generate from Maven archetype

### LLM CLI Server (`llm-cli`)

- `list_available_clis` - List available LLM CLI tools
- `execute_gemini_cli` - Execute Gemini CLI commands
- `execute_codex_cli` - Execute Codex CLI commands
- `execute_openai_cli` - Execute OpenAI CLI commands
- `gemini_chat` - Chat with Gemini
- `codex_generate` - Generate with Codex
- `openai_chat` - Chat with OpenAI

### Documentation Server (`langchain-docs`, `langgraph-docs`)

- `search_langchain_docs` - Search LangChain documentation
- `search_langgraph_docs` - Search LangGraph documentation
- `get_langchain_examples` - Get LangChain code examples by category
- `get_langgraph_examples` - Get LangGraph code examples by category
- `list_available_docs` - List all available documentation sources

## Troubleshooting

### Server won't start

1. Check that MCP SDK is installed: `pip install mcp`
2. Verify Python path: `python --version`
3. Check for import errors: `python -c "from langgang import mcp_server_runner"`

### Tools not appearing in client

1. Verify `mcp-config.json` is correctly formatted
2. Check that the server process is running
3. Review client logs for connection errors

### Connection issues

1. Ensure stdio communication is enabled in your MCP client
2. Check that no other process is using the stdio streams
3. Verify the command and args in `mcp-config.json` are correct
