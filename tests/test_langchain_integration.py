"""
Tests for LangChain integration
"""

from langgang.langchain_integration import (
    TemplateGenerationTool,
    ListTemplatesTool,
    get_langgang_tools
)


def test_list_templates_tool():
    """Test ListTemplatesTool."""
    tool = ListTemplatesTool()
    
    assert tool.name == "list_templates"
    assert tool.description is not None
    
    result = tool._run()
    assert isinstance(result, str)
    assert "Available Templates" in result


def test_template_generation_tool():
    """Test TemplateGenerationTool initialization."""
    tool = TemplateGenerationTool()
    
    assert tool.name == "generate_template"
    assert tool.description is not None
    assert tool.args_schema is not None


def test_get_langgang_tools():
    """Test getting all LangGang tools."""
    tools = get_langgang_tools()
    
    assert isinstance(tools, list)
    assert len(tools) == 2
    
    tool_names = [tool.name for tool in tools]
    assert "list_templates" in tool_names
    assert "generate_template" in tool_names
