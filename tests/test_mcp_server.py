"""
Tests for LangGang MCP server
"""

import pytest
from pathlib import Path
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
