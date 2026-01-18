"""
Tests for documentation MCP server
"""

from langgang.docs_mcp_server import DocumentationMCPServer


def test_docs_server_initialization():
    """Test documentation server initializes correctly."""
    server = DocumentationMCPServer()
    
    assert server.langchain_docs_url is not None
    assert server.langgraph_docs_url is not None
    assert len(server.langchain_topics) > 0
    assert len(server.langgraph_topics) > 0


def test_search_langchain_docs():
    """Test searching LangChain docs."""
    server = DocumentationMCPServer()
    result = server.search_langchain_docs("agents")
    
    assert isinstance(result, dict)
    assert result["source"] == "langchain"
    assert "query" in result
    assert "base_url" in result


def test_search_langgraph_docs():
    """Test searching LangGraph docs."""
    server = DocumentationMCPServer()
    result = server.search_langgraph_docs("graphs")
    
    assert isinstance(result, dict)
    assert result["source"] == "langgraph"
    assert "query" in result
    assert "base_url" in result


def test_get_langchain_examples():
    """Test getting LangChain examples."""
    server = DocumentationMCPServer()
    result = server.get_langchain_examples("chains")
    
    assert isinstance(result, dict)
    assert "examples" in result
    assert "chains" in result["examples"]


def test_get_langgraph_examples():
    """Test getting LangGraph examples."""
    server = DocumentationMCPServer()
    result = server.get_langgraph_examples("graphs")
    
    assert isinstance(result, dict)
    assert "examples" in result
    assert "graphs" in result["examples"]


def test_list_available_docs():
    """Test listing available documentation."""
    server = DocumentationMCPServer()
    docs = server.list_available_docs()
    
    assert isinstance(docs, dict)
    assert "langchain" in docs
    assert "langgraph" in docs
    assert "url" in docs["langchain"]
    assert "topics" in docs["langchain"]
