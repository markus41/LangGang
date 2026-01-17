# GitHub Copilot Configuration for LangGang

Welcome to LangGang! This guide helps GitHub Copilot provide better assistance when working with this AI-powered code templating framework.

## Repository Overview

LangGang integrates multiple templating systems and AI frameworks:
- **LangChain** & **LangGraph** for AI-powered workflows
- **Cookiecutter**, **Copier**, and **Maven Archetypes** for code templating
- **MCP (Model Context Protocol)** for AI assistant integration
- **Harness CI/CD** for automated pipelines

## Key Technologies

### Python Stack
- **Python 3.9+** required
- **LangChain** (>=0.1.0) - LLM application framework
- **LangGraph** (>=0.0.20) - Stateful workflows
- **Cookiecutter** (>=2.5.0) - Python templating
- **Copier** (>=9.0.0) - Modern templating
- **Pydantic** (>=2.0.0) - Data validation

### Java Stack
- **Maven 3.x** for archetype management
- **Java 11+** for generated projects
- **LangChain4j** for Java LangChain implementation

### Template Systems
1. **Cookiecutter**: Uses `cookiecutter.json` and Jinja2 (`{{ variable }}`)
2. **Copier**: Uses `copier.yml` and Jinja2 with better updating
3. **Maven**: Uses `archetype-metadata.xml` and Maven placeholders (`${variable}`)

## Project Structure

```
LangGang/
├── langgang/                    # Python package
│   ├── mcp_server.py           # Template MCP server
│   ├── docs_mcp_server.py      # Documentation MCP server
│   ├── langchain_integration.py # LangChain tools
│   └── langgraph_integration.py # LangGraph workflows
├── templates/                   # Template collections
│   ├── cookiecutter/           # Python-based templates
│   ├── copier/                 # Modern templates
│   └── maven/                  # Java archetypes
├── .claude/                    # Claude Code configs
├── .copilot/                   # GitHub Copilot configs
├── .harness/                   # CI/CD pipelines
├── pyproject.toml              # Python package config
└── requirements.txt            # Python dependencies
```

## Coding Conventions

### Python Code Style
```python
# Use type hints everywhere
def function_name(param: str, optional: Optional[int] = None) -> Dict[str, Any]:
    """Docstring with Google style.
    
    Args:
        param: Description
        optional: Optional parameter
        
    Returns:
        Dictionary with results
    """
    pass

# Pydantic models for schemas
from pydantic import BaseModel, Field

class MySchema(BaseModel):
    """Schema description."""
    field_name: str = Field(description="Field description")
```

### LangChain Patterns
```python
# Tool creation
from langchain_core.tools import BaseTool

class MyTool(BaseTool):
    name = "tool_name"
    description = "Tool description"
    
    def _run(self, param: str) -> str:
        """Execute the tool."""
        return result

# Chain creation
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("template")
chain = prompt | llm | output_parser
```

### LangGraph Patterns
```python
# State definition
from typing import Annotated, TypedDict
import operator

class MyState(TypedDict):
    messages: Annotated[list, operator.add]
    data: str

# Graph creation
from langgraph.graph import StateGraph, END

workflow = StateGraph(MyState)
workflow.add_node("process", process_function)
workflow.set_entry_point("process")
workflow.add_edge("process", END)
graph = workflow.compile()
```

### MCP Server Patterns
```python
# MCP tool method
def tool_name(self, param: str) -> Dict[str, Any]:
    """Tool description.
    
    Args:
        param: Parameter description
        
    Returns:
        Success: {"status": "success", "data": result}
        Error: {"error": "error message"}
    """
    try:
        result = process(param)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": str(e)}
```

## Template Development

### Cookiecutter Templates
```
templates/cookiecutter/my-template/
├── cookiecutter.json              # Configuration
└── {{cookiecutter.project_slug}}/ # Template directory
    ├── README.md                  # Project README
    ├── __init__.py               # Python package
    └── requirements.txt          # Dependencies
```

Configuration format:
```json
{
  "project_name": "My Project",
  "project_slug": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
  "author": "Author Name"
}
```

### Copier Templates
```
templates/copier/my-template/
├── copier.yml                # Configuration
└── template/                 # Template directory
    ├── README.md.jinja
    └── {{project_slug}}/
        └── __init__.py.jinja
```

Configuration format:
```yaml
project_name:
  type: str
  help: Project name
  default: My Project

use_feature:
  type: bool
  help: Include feature?
  default: yes
```

### Maven Archetypes
```
templates/maven/my-archetype/
├── pom.xml
└── src/main/resources/
    ├── META-INF/maven/
    │   └── archetype-metadata.xml
    └── archetype-resources/
        ├── pom.xml
        └── src/main/java/
            └── App.java
```

## Common Tasks

### Adding a New Template
1. Choose template type (Cookiecutter/Copier/Maven)
2. Create directory structure
3. Add configuration file
4. Create template files with placeholders
5. Test generation

### Creating a LangChain Tool
1. Define Pydantic input schema
2. Create tool class inheriting from BaseTool
3. Implement _run method
4. Add to get_langgang_tools()
5. Write tests

### Building a LangGraph Workflow
1. Define TypedDict state
2. Create StateGraph
3. Implement node functions
4. Add nodes and edges
5. Compile graph

### Extending MCP Server
1. Add method to MCP server class
2. Update mcp-config.json capabilities
3. Add error handling and logging
4. Document the tool

## Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run linters
black langgang/
ruff check langgang/
mypy langgang/

# Test MCP server
python -m langgang.mcp_server
```

## Documentation Resources

- [LangChain Docs](https://python.langchain.com/docs)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph)
- [Cookiecutter Docs](https://cookiecutter.readthedocs.io)
- [Copier Docs](https://copier.readthedocs.io)
- [Maven Archetypes](https://maven.apache.org/guides/introduction/introduction-to-archetypes.html)
- [MCP Protocol](https://modelcontextprotocol.io)

## Copilot Tips

1. **Template Variables**: Use correct placeholder syntax for each type
   - Cookiecutter: `{{ cookiecutter.variable }}`
   - Copier: `{{ variable }}`
   - Maven: `${variable}`

2. **Type Hints**: Always include type hints in Python code

3. **Pydantic Models**: Use for all schema definitions

4. **Error Handling**: Return proper error formats in MCP tools

5. **Documentation**: Add comprehensive docstrings

6. **Testing**: Write tests for new functionality

## Best Practices

- Keep templates focused and modular
- Provide sensible defaults in template configs
- Add comprehensive README templates
- Use proper error handling in MCP servers
- Follow LangChain/LangGraph patterns
- Document all public APIs
- Test template generation end-to-end
