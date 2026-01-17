"""
Tests for LangGang MCP server
"""

from langgang.mcp_server import LangGangMCPServer


def test_mcp_server_initialization():
    """Test MCP server initializes correctly."""
    server = LangGangMCPServer()
    
    assert server.templates_dir is not None
    assert server.cookiecutter_dir is not None
    assert server.copier_dir is not None
    assert server.maven_dir is not None


def test_list_templates():
    """Test listing templates."""
    server = LangGangMCPServer()
    templates = server.list_templates()
    
    assert isinstance(templates, dict)
    assert "cookiecutter" in templates
    assert "copier" in templates
    assert "maven" in templates
    
    # Check for our example templates
    if templates["cookiecutter"]:
        assert "python-langchain-project" in templates["cookiecutter"]
    if templates["copier"]:
        assert "langgraph-agent" in templates["copier"]
    if templates["maven"]:
        assert "langchain-java-archetype" in templates["maven"]


def test_list_templates_returns_lists():
    """Test that list_templates returns lists for each category."""
    server = LangGangMCPServer()
    templates = server.list_templates()
    
    assert isinstance(templates["cookiecutter"], list)
    assert isinstance(templates["copier"], list)
    assert isinstance(templates["maven"], list)


def test_generate_from_nonexistent_template():
    """Test error handling when template doesn't exist."""
    server = LangGangMCPServer()
    
    # Test Cookiecutter
    result = server.generate_from_cookiecutter(
        "nonexistent-template",
        "/tmp/test",
        {}
    )
    assert "error" in result
    assert "not found" in result["error"].lower()
    
    # Test Copier
    result = server.generate_from_copier(
        "nonexistent-template",
        "/tmp/test",
        {}
    )
    assert "error" in result
    assert "not found" in result["error"].lower()
    
    # Test Maven
    result = server.generate_from_maven_archetype(
        "nonexistent-archetype",
        "/tmp/test",
        {}
    )
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_path_traversal_prevention():
    """Test that path traversal attacks are prevented."""
    server = LangGangMCPServer()
    
    # Test with directory traversal in template name
    result = server.generate_from_cookiecutter(
        "../../../etc/passwd",
        "/tmp/test",
        {}
    )
    assert "error" in result
    assert "must not contain" in result["error"] or "Invalid" in result["error"]
    
    # Test with absolute path
    result = server.generate_from_copier(
        "/etc/passwd",
        "/tmp/test",
        {}
    )
    assert "error" in result


def test_maven_command_injection_prevention():
    """Test that command injection is prevented in Maven properties."""
    server = LangGangMCPServer()
    
    # Test with shell metacharacters in property values
    result = server.generate_from_maven_archetype(
        "test-archetype",
        "/tmp/test",
        {"groupId": "com.test; rm -rf /"}
    )
    assert "error" in result
    assert "invalid" in result["error"].lower()

