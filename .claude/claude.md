# Claude Code Configuration for LangGang

Welcome to the LangGang repository! This file provides guidance for Claude Code when working with this AI-powered code templating framework.

## Repository Overview

LangGang is an AI-powered code templating framework that integrates:
- **LangChain**: Framework for building LLM applications
- **LangGraph**: Stateful multi-actor application framework
- **Cookiecutter**: Python-based project templating
- **Copier**: Modern project templating tool
- **Maven Archetypes**: Java project templating
- **MCP (Model Context Protocol)**: For AI assistant integration
- **Harness**: CI/CD pipeline configuration

## Project Structure

```
LangGang/
├── .claude/              # Claude Code configurations
│   ├── agents/          # Custom agent definitions
│   ├── mcps/            # MCP server configurations
│   ├── skills/          # Reusable skills for Claude
│   └── slash-commands/  # Custom slash commands
├── .copilot/            # GitHub Copilot configurations
├── langgang/            # Main Python package
│   ├── __init__.py
│   ├── mcp_server.py            # Template generation MCP server
│   ├── docs_mcp_server.py       # Documentation MCP server
│   ├── langchain_integration.py # LangChain tools
│   └── langgraph_integration.py # LangGraph workflows
├── templates/           # Template collections
│   ├── cookiecutter/   # Cookiecutter templates
│   ├── copier/         # Copier templates
│   └── maven/          # Maven archetypes
├── .harness/           # Harness CI/CD pipelines
└── tests/              # Test suite

```

## Key Concepts

### 1. Template Types

**Cookiecutter Templates**
- Python-based templating using Jinja2
- Located in `templates/cookiecutter/`
- Use `cookiecutter.json` for configuration
- Best for: Python projects, data science, ML applications

**Copier Templates**
- Modern templating with better updating support
- Located in `templates/copier/`
- Use `copier.yml` for configuration
- Best for: Projects that need to stay in sync with templates

**Maven Archetypes**
- Java project templating
- Located in `templates/maven/`
- Standard Maven archetype structure
- Best for: Java/JVM projects, enterprise applications

### 2. MCP Integration

The repository includes MCP servers for:
- Template generation (`langgang.mcp_server`)
- Documentation access (`langgang.docs_mcp_server`)

MCP servers enable AI assistants to:
- Generate code from templates
- Access LangChain/LangGraph documentation
- List available templates
- Get code examples

### 3. LangChain Integration

The `langchain_integration.py` module provides:
- `TemplateGenerationTool`: LangChain tool for generating code
- `ListTemplatesTool`: Tool for discovering templates
- `create_template_generation_prompt()`: Prompt template for AI assistance

### 4. LangGraph Workflows

The `langgraph_integration.py` module provides:
- Multi-step template generation workflows
- State management for complex scaffolding
- Nodes for: requirement analysis, template selection, context gathering, code generation

## Development Guidelines

### When Adding New Templates

1. Choose the appropriate template type (Cookiecutter, Copier, or Maven)
2. Create template directory structure
3. Add template configuration file
4. Create template files with placeholders
5. Test template generation
6. Document template usage

### When Working with MCP Servers

1. MCP servers are in the `langgang/` package
2. Use Python type hints and docstrings
3. Follow the MCP protocol specification
4. Test with MCP client tools

### Code Style

- Python: Follow PEP 8, use Black formatter, type hints required
- Use Ruff for linting
- Keep functions focused and small
- Write docstrings for all public APIs

### Testing

- Add tests in `tests/` directory
- Use pytest for Python tests
- Test template generation end-to-end
- Mock external dependencies

## Common Tasks

### Generate Code from Template

```python
from langgang.mcp_server import LangGangMCPServer

server = LangGangMCPServer()
result = server.generate_from_cookiecutter(
    template_name="python-langchain-project",
    output_dir="/path/to/output",
    context={"project_name": "My Project"}
)
```

### Create LangChain Agent

```python
from langgang.langchain_integration import get_langgang_tools
from langchain.agents import AgentExecutor

tools = get_langgang_tools()
# Create agent with tools
```

### Run LangGraph Workflow

```python
from langgang.langgraph_integration import create_template_generation_graph

graph = create_template_generation_graph()
result = graph.invoke({"project_description": "Python Flask API"})
```

## Slash Commands

LangGang provides custom slash commands for Claude Code. All commands are defined in `.claude/slash-commands/`:

### Template Commands
- **`/list-templates [type]`** - List all available templates, optionally filtered by type
- **`/generate-template <type> <name> [options]`** - Generate code from a template
- **`/create-template <type> <name>`** - Create a new template

### Documentation Commands
- **`/search-docs <framework> <query>`** - Search LangChain or LangGraph documentation
- **`/get-examples <framework> [category]`** - Get code examples from LangChain or LangGraph

### LLM CLI Commands (NEW!)
- **`/gemini-chat <prompt> [options]`** - Chat with Google Gemini using CLI
- **`/codex-generate <prompt> [options]`** - Generate code using Codex CLI
- **`/openai-chat <prompt> [options]`** - Chat with OpenAI using CLI
- **`/list-clis`** - List available LLM CLI tools

### Development Commands
- **`/new-langchain-tool <name> <description>`** - Create a new LangChain tool
- **`/new-mcp-tool <name> <description>`** - Create a new MCP tool
- **`/run-test [path]`** - Run tests with pytest
- **`/format-code [path]`** - Format code with black and ruff
- **`/type-check [path]`** - Run mypy type checking
- **`/check-health`** - Check repository health

### Usage Examples

```
/list-templates
/generate-template cookiecutter python-langchain-project
/gemini-chat Explain LangChain --model=gemini-pro
/codex-generate Create a Python function --model=gpt-4
/openai-chat What is MCP? --model=gpt-4
/list-clis
/search-docs langchain agents
/get-examples langgraph graphs
/run-test
/format-code
/type-check
/check-health
```

All LLM CLI commands require the respective CLI tools to be installed and configured.

## Documentation Resources

- [LangChain Documentation](https://python.langchain.com/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph)
- [Cookiecutter Documentation](https://cookiecutter.readthedocs.io)
- [Copier Documentation](https://copier.readthedocs.io)
- [Maven Archetypes](https://maven.apache.org/guides/introduction/introduction-to-archetypes.html)
- [MCP Protocol](https://modelcontextprotocol.io)

## Claude Code Capabilities

As Claude Code, you can:
- Generate new templates using the patterns in this repo
- Create LangChain tools for template generation
- Build LangGraph workflows for complex scaffolding
- Write MCP server code
- Add new template types
- Improve documentation
- Fix bugs and add tests

## Tips for Claude

1. **Template Generation**: Always validate template syntax before committing
2. **MCP Servers**: Test MCP endpoints with the MCP inspector
3. **LangChain/LangGraph**: Follow the official documentation patterns
4. **Dependencies**: Check compatibility before adding new packages
5. **CI/CD**: Harness pipelines are in `.harness/` directory

## Getting Help

- Check the MCP server logs for debugging
- Use the documentation MCP server for LangChain/LangGraph help
- Review existing templates for patterns
- Test changes with the Harness CI/CD pipeline
