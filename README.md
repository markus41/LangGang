# LangGang

**AI-Powered Code Templating Framework**

LangGang combines the power of LangChain, LangGraph, Cookiecutter, Copier, and Maven Archetypes to create an intelligent code scaffolding system. Transform developer workflows with AI-assisted template generation and multi-framework integration.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Multiple Template Systems**: Support for Cookiecutter, Copier, and Maven Archetypes
- **LangChain Integration**: AI-powered template selection and generation
- **LangGraph Workflows**: Multi-step code generation with state management
- **MCP Protocol**: Model Context Protocol servers for AI assistant integration
- **Documentation Access**: Built-in MCP servers for LangChain/LangGraph docs
- **Harness CI/CD**: Automated pipeline configuration
- **Claude Code & Copilot**: Enhanced IDE integration with agents and skills

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Template Types](#template-types)
- [MCP Integration](#mcp-integration)
- [LangChain & LangGraph](#langchain--langgraph)
- [AI Assistant Integration](#ai-assistant-integration)
- [Development](#development)
- [Documentation](#documentation)
- [Contributing](#contributing)

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- Maven 3.x (for Maven archetypes)
- Git

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/markus41/LangGang.git
cd LangGang

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"
```

## 🏃 Quick Start

### Using MCP Server

```python
from langgang.mcp_server import LangGangMCPServer

# Initialize server
server = LangGangMCPServer()

# List available templates
templates = server.list_templates()
print(templates)

# Generate from Cookiecutter template
result = server.generate_from_cookiecutter(
    template_name="python-langchain-project",
    output_dir="./my-project",
    context={
        "project_name": "My AI App",
        "author": "Your Name",
        "use_langchain": "y"
    }
)
```

### Using LangChain Tools

```python
from langgang.langchain_integration import get_langgang_tools

# Get LangChain tools for template generation
tools = get_langgang_tools()

# Use with LangChain agents
from langchain.agents import AgentExecutor
# ... create agent with tools
```

### Using LangGraph Workflows

```python
from langgang.langgraph_integration import create_template_generation_graph

# Create workflow graph
graph = create_template_generation_graph()

# Run workflow
result = graph.invoke({
    "project_description": "Python Flask API with authentication"
})
```

## 📦 Template Types

### Cookiecutter Templates

Located in `templates/cookiecutter/`

**Features:**
- Python-based templating with Jinja2
- Rich filter support
- Best for Python projects and data science

**Example:**
```bash
# Using cookiecutter CLI
cookiecutter templates/cookiecutter/python-langchain-project/

# Using LangGang
python -c "from langgang.mcp_server import LangGangMCPServer; \
server = LangGangMCPServer(); \
server.generate_from_cookiecutter('python-langchain-project', './output', {})"
```

### Copier Templates

Located in `templates/copier/`

**Features:**
- Modern templating with update support
- YAML configuration
- Best for projects that evolve with templates

**Example:**
```bash
# Using copier CLI
copier copy templates/copier/langgraph-agent/ ./my-agent

# Using LangGang
python -c "from langgang.mcp_server import LangGangMCPServer; \
server = LangGangMCPServer(); \
server.generate_from_copier('langgraph-agent', './output', {})"
```

### Maven Archetypes

Located in `templates/maven/`

**Features:**
- Java/JVM project templating
- Standard Maven conventions
- Best for enterprise Java applications

**Example:**
```bash
# Using Maven CLI
mvn archetype:generate \
  -DarchetypeGroupId=com.langgang \
  -DarchetypeArtifactId=langchain-java-archetype

# Using LangGang
python -c "from langgang.mcp_server import LangGangMCPServer; \
server = LangGangMCPServer(); \
server.generate_from_maven_archetype('langchain-java-archetype', './output', {})"
```

## 🔌 MCP Integration

LangGang provides Model Context Protocol (MCP) servers for AI assistant integration.

### Available MCP Servers

1. **Template Generation Server** (`langgang.mcp_server`)
   - Generate code from templates
   - List available templates
   - Support for all template types

2. **Documentation Server** (`langgang.docs_mcp_server`)
   - Search LangChain documentation
   - Search LangGraph documentation
   - Get code examples
   - List available documentation

### Configuration

MCP servers are configured in `mcp-config.json`:

```json
{
  "mcpServers": {
    "langgang-templates": {
      "command": "python",
      "args": ["-m", "langgang.mcp_server"]
    },
    "langchain-docs": {
      "command": "python",
      "args": ["-m", "langgang.docs_mcp_server"]
    }
  }
}
```

### Running MCP Servers

```bash
# Start template generation server
python -m langgang.mcp_server

# Start documentation server
python -m langgang.docs_mcp_server
```

## 🤖 LangChain & LangGraph

### LangChain Tools

LangGang provides LangChain tools for template generation:

```python
from langgang.langchain_integration import (
    TemplateGenerationTool,
    ListTemplatesTool,
    get_langgang_tools
)

# Use individual tools
list_tool = ListTemplatesTool()
result = list_tool._run()

gen_tool = TemplateGenerationTool()
result = gen_tool._run(
    template_type="cookiecutter",
    template_name="python-langchain-project",
    output_directory="./output",
    context={"project_name": "My App"}
)

# Or get all tools at once
tools = get_langgang_tools()
```

### LangGraph Workflows

Multi-step template generation workflows:

```python
from langgang.langgraph_integration import create_template_generation_graph

# Create workflow
graph = create_template_generation_graph()

# Execute workflow
result = graph.invoke({
    "project_description": "FastAPI application with authentication",
    "context_variables": {
        "project_name": "my_api",
        "author": "Developer"
    }
})
```

**Workflow Steps:**
1. Analyze requirements
2. Select appropriate template
3. Gather context variables
4. Generate code
5. Validate output

## 🎯 AI Assistant Integration

### Claude Code Integration

LangGang includes extensive Claude Code configurations in `.claude/`:

- **Agents**: Specialized agents for template generation, LangChain development, MCP development, and CI/CD
- **MCPs**: MCP server configurations
- **Skills**: Reusable skills for common tasks
- **Slash Commands**: Quick commands for template operations

**Using Claude Code:**
```bash
# Claude Code automatically loads configurations from .claude/
# See .claude/claude.md for full documentation
```

### GitHub Copilot Integration

Copilot configurations in `.copilot/`:

- **Agents**: Context-aware coding assistants
- **Extensions**: Workspace-specific Copilot extensions
- **Code Patterns**: Templates and snippets

**Using GitHub Copilot:**
```bash
# Copilot automatically uses configurations from .copilot/
# See .copilot/copilot.md for full documentation
```

## 🛠️ Development

### Project Structure

```
LangGang/
├── .claude/              # Claude Code configurations
│   ├── agents/          # Specialized agents
│   ├── mcps/            # MCP configurations
│   ├── skills/          # Reusable skills
│   ├── slash-commands/  # Quick commands
│   └── claude.md        # Claude documentation
├── .copilot/            # GitHub Copilot configurations
│   ├── agents/          # Copilot agents
│   ├── extensions/      # Copilot extensions
│   └── copilot.md       # Copilot documentation
├── .harness/            # Harness CI/CD pipelines
├── langgang/            # Main Python package
│   ├── __init__.py
│   ├── mcp_server.py
│   ├── docs_mcp_server.py
│   ├── langchain_integration.py
│   └── langgraph_integration.py
├── templates/           # Template collections
│   ├── cookiecutter/
│   ├── copier/
│   └── maven/
└── tests/               # Test suite
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=langgang --cov-report=html

# Run specific test file
pytest tests/test_mcp_server.py -v
```

### Code Quality

```bash
# Format code
black langgang/

# Lint code
ruff check langgang/

# Type checking
mypy langgang/
```

### CI/CD

LangGang uses Harness for CI/CD. Pipeline configuration is in `.harness/`:

- `pipeline.yaml`: Main CI/CD pipeline
- `trigger.yaml`: PR and push triggers

## 📚 Documentation

### Official Documentation

- [LangChain Documentation](https://python.langchain.com/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph)
- [Cookiecutter Documentation](https://cookiecutter.readthedocs.io)
- [Copier Documentation](https://copier.readthedocs.io)
- [Maven Archetypes](https://maven.apache.org/guides/introduction/introduction-to-archetypes.html)
- [MCP Protocol](https://modelcontextprotocol.io)

### LangGang Documentation

- `.claude/claude.md`: Claude Code integration guide
- `.copilot/copilot.md`: GitHub Copilot integration guide
- API documentation: Generated from docstrings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and linting
6. Submit a pull request

### Coding Standards

- Follow PEP 8 for Python code
- Use type hints everywhere
- Add docstrings to all public APIs
- Write tests for new features
- Keep commits focused and atomic

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- Cookiecutter and Copier communities
- Maven project for archetype system
- Model Context Protocol creators

## 📧 Contact

- GitHub: [@markus41](https://github.com/markus41)
- Repository: [LangGang](https://github.com/markus41/LangGang)

---

**Made with ❤️ by the LangGang community**