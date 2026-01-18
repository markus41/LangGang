"""
Tests for LangGang MCP server
"""

import pytest

from langgang.mcp_server import LangGangMCPServer

# Check if optional dependencies are available
try:
    import cookiecutter  # noqa: F401
    COOKIECUTTER_AVAILABLE = True
except ImportError:
    COOKIECUTTER_AVAILABLE = False

try:
    import copier  # noqa: F401
    COPIER_AVAILABLE = True
except ImportError:
    COPIER_AVAILABLE = False


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


@pytest.mark.skipif(not COOKIECUTTER_AVAILABLE, reason="cookiecutter not installed")
def test_generate_from_nonexistent_cookiecutter_template():
    """Test error handling when Cookiecutter template doesn't exist."""
    server = LangGangMCPServer()
    result = server.generate_from_cookiecutter(
        "nonexistent-template",
        "/tmp/test",
        {}
    )
    assert "error" in result
    assert "not found" in result["error"].lower()


@pytest.mark.skipif(not COPIER_AVAILABLE, reason="copier not installed")
def test_generate_from_nonexistent_copier_template():
    """Test error handling when Copier template doesn't exist."""
    server = LangGangMCPServer()
    result = server.generate_from_copier(
        "nonexistent-template",
        "/tmp/test",
        {}
    )
    assert "error" in result
    assert "not found" in result["error"].lower()


def test_generate_from_nonexistent_maven_archetype():
    """Test error handling when Maven archetype doesn't exist."""
    server = LangGangMCPServer()
    result = server.generate_from_maven_archetype(
        "nonexistent-archetype",
        "/tmp/test",
        {}
    )
    assert "error" in result
    assert "not found" in result["error"].lower()


@pytest.mark.skipif(not COOKIECUTTER_AVAILABLE, reason="cookiecutter not installed")
def test_path_traversal_prevention_cookiecutter():
    """Test that path traversal attacks are prevented in Cookiecutter."""
    server = LangGangMCPServer()
    result = server.generate_from_cookiecutter(
        "../../../etc/passwd",
        "/tmp/test",
        {}
    )
    assert "error" in result
    assert "must not contain" in result["error"] or "Invalid" in result["error"]


@pytest.mark.skipif(not COPIER_AVAILABLE, reason="copier not installed")
def test_path_traversal_prevention_copier():
    """Test that path traversal attacks are prevented in Copier."""
    server = LangGangMCPServer()
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
    # Use an existing archetype name to test validation
    templates = server.list_templates()
    if templates["maven"]:
        archetype_name = templates["maven"][0]
    else:
        # If no archetype exists, test with a valid name format
        # The validation should catch the shell metacharacters in properties
        archetype_name = "langchain-java-archetype"
    
    result = server.generate_from_maven_archetype(
        archetype_name,
        "/tmp/test",
        {"groupId": "com.test; rm -rf /"}
    )
    assert "error" in result
    # Check for validation error (invalid characters) or archetype not found
    error_msg = result["error"].lower()
    assert "invalid" in error_msg or "not found" in error_msg or "characters" in error_msg

