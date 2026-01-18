"""
Tests for template metadata and discovery system.

Tests cover:
- TemplateMetadata class
- Metadata loading and inference
- Template search and filtering
- Relevance scoring
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from langgang.mcp_server import (
    LangGangMCPServer,
    TemplateMetadata,
    _sanitize_property_value,
    _validate_path,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_metadata():
    """Create sample template metadata."""
    return TemplateMetadata(
        name="python-langchain-project",
        display_name="Python LangChain Project",
        description="Production-ready Python project with LangChain",
        version="1.0.0",
        author="LangGang",
        tags=["python", "langchain", "ai"],
        category="backend",
        difficulty="beginner",
        frameworks=["LangChain", "Pydantic"],
        languages=["Python"],
        prerequisites=["python>=3.9"],
        features=["LangChain integration", "Async support"],
        compatible_with=["copier/langgraph-agent"],
        template_type="cookiecutter",
    )


@pytest.fixture
def temp_templates_dir():
    """Create a temporary templates directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        templates_dir = Path(tmpdir)

        # Create cookiecutter template
        cc_dir = templates_dir / "cookiecutter" / "test-template"
        cc_dir.mkdir(parents=True)
        (cc_dir / "cookiecutter.json").write_text(json.dumps({
            "project_name": "My Project",
            "author": "Test Author",
            "python_version": "3.11",
            "use_langchain": "y",
        }))

        # Create copier template
        cp_dir = templates_dir / "copier" / "test-agent"
        cp_dir.mkdir(parents=True)
        (cp_dir / "copier.yml").write_text("""
project_name:
  type: str
  default: "Test Project"
use_langchain:
  type: bool
  default: true
python_version:
  type: str
  default: "3.11"
""")

        # Create maven template
        mv_dir = templates_dir / "maven" / "test-archetype"
        mv_dir.mkdir(parents=True)
        (mv_dir / "pom.xml").write_text("<project></project>")

        yield templates_dir


@pytest.fixture
def server_with_temp_templates(temp_templates_dir):
    """Create server with temporary templates."""
    return LangGangMCPServer(templates_dir=temp_templates_dir)


# ============================================================================
# TemplateMetadata Tests
# ============================================================================

class TestTemplateMetadata:
    """Tests for TemplateMetadata class."""

    def test_to_dict(self, sample_metadata):
        """Test conversion to dictionary."""
        result = sample_metadata.to_dict()

        assert result["name"] == "python-langchain-project"
        assert result["schema_version"] == "1.0"
        assert "LangChain" in result["frameworks"]
        assert result["difficulty"] == "beginner"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "name": "test-template",
            "display_name": "Test Template",
            "description": "A test template",
            "tags": ["test", "demo"],
            "category": "testing",
        }

        result = TemplateMetadata.from_dict(data)

        assert result.name == "test-template"
        assert result.display_name == "Test Template"
        assert "test" in result.tags

    def test_default_display_name(self):
        """Test automatic display name generation."""
        metadata = TemplateMetadata(name="my-cool-template")

        assert metadata.display_name == "My Cool Template"

    def test_matches_query_text(self, sample_metadata):
        """Test text query matching."""
        # Exact name match
        score = sample_metadata.matches_query(query="python-langchain")
        assert score > 0.4

        # Partial word match
        score = sample_metadata.matches_query(query="python")
        assert score > 0.2

        # No match
        score = sample_metadata.matches_query(query="java spring")
        assert score < 0.3

    def test_matches_query_tags(self, sample_metadata):
        """Test tag matching."""
        score = sample_metadata.matches_query(tags=["python", "ai"])
        assert score > 0.5

        score = sample_metadata.matches_query(tags=["java"])
        assert score == 0.0

    def test_matches_query_category(self, sample_metadata):
        """Test category matching."""
        score = sample_metadata.matches_query(category="backend")
        assert score == 1.0

        score = sample_metadata.matches_query(category="frontend")
        assert score == 0.0

    def test_matches_query_frameworks(self, sample_metadata):
        """Test framework matching."""
        score = sample_metadata.matches_query(frameworks=["LangChain"])
        assert score > 0.5

        score = sample_metadata.matches_query(frameworks=["Django", "Flask"])
        assert score == 0.0

    def test_matches_query_languages(self, sample_metadata):
        """Test language matching."""
        score = sample_metadata.matches_query(languages=["Python"])
        assert score > 0.5

        score = sample_metadata.matches_query(languages=["Java"])
        assert score == 0.0

    def test_matches_query_combined(self, sample_metadata):
        """Test combined criteria matching."""
        score = sample_metadata.matches_query(
            query="python",
            tags=["ai"],
            category="backend",
            frameworks=["LangChain"],
        )
        assert score > 0.7


# ============================================================================
# LangGangMCPServer Tests
# ============================================================================

class TestLangGangMCPServerDiscovery:
    """Tests for template discovery functionality."""

    def test_list_templates(self, server_with_temp_templates):
        """Test listing all templates."""
        templates = server_with_temp_templates.list_templates()

        assert "cookiecutter" in templates
        assert "copier" in templates
        assert "maven" in templates
        assert "test-template" in templates["cookiecutter"]
        assert "test-agent" in templates["copier"]
        assert "test-archetype" in templates["maven"]

    def test_list_templates_with_metadata(self, server_with_temp_templates):
        """Test listing templates with metadata."""
        result = server_with_temp_templates.list_templates_with_metadata()

        assert "cookiecutter" in result
        assert len(result["cookiecutter"]) > 0
        assert "name" in result["cookiecutter"][0]
        assert "metadata" in result["cookiecutter"][0]

    def test_get_template_metadata_from_file(self, temp_templates_dir):
        """Test loading metadata from template-meta.json."""
        # Create metadata file
        meta_file = temp_templates_dir / "cookiecutter" / "test-template" / "template-meta.json"
        meta_file.write_text(json.dumps({
            "name": "test-template",
            "display_name": "Test Template",
            "description": "A test template with metadata",
            "tags": ["test", "meta"],
            "category": "testing",
        }))

        server = LangGangMCPServer(templates_dir=temp_templates_dir)
        metadata = server.get_template_metadata("cookiecutter", "test-template")

        assert metadata["display_name"] == "Test Template"
        assert metadata["description"] == "A test template with metadata"
        assert "test" in metadata["tags"]

    def test_get_template_metadata_inferred(self, server_with_temp_templates):
        """Test metadata inference from config files."""
        metadata = server_with_temp_templates.get_template_metadata(
            "cookiecutter", "test-template"
        )

        assert metadata["name"] == "test-template"
        assert "Python" in metadata["languages"]
        assert "LangChain" in metadata["frameworks"]

    def test_get_template_metadata_caching(self, server_with_temp_templates):
        """Test that metadata is cached."""
        # First call
        metadata1 = server_with_temp_templates.get_template_metadata(
            "cookiecutter", "test-template"
        )
        # Second call (should use cache)
        metadata2 = server_with_temp_templates.get_template_metadata(
            "cookiecutter", "test-template"
        )

        assert metadata1 == metadata2
        assert "cookiecutter/test-template" in server_with_temp_templates._metadata_cache


class TestLangGangMCPServerVariables:
    """Tests for template variable extraction."""

    def test_get_cookiecutter_variables(self, server_with_temp_templates):
        """Test extracting variables from Cookiecutter template."""
        variables = server_with_temp_templates.get_template_variables(
            "cookiecutter", "test-template"
        )

        assert "project_name" in variables
        assert "author" in variables
        assert "python_version" in variables

    def test_get_copier_variables(self, server_with_temp_templates):
        """Test extracting variables from Copier template."""
        variables = server_with_temp_templates.get_template_variables(
            "copier", "test-agent"
        )

        assert "project_name" in variables
        assert "use_langchain" in variables

    def test_get_maven_variables(self, server_with_temp_templates):
        """Test default Maven archetype variables."""
        variables = server_with_temp_templates.get_template_variables(
            "maven", "test-archetype"
        )

        assert "groupId" in variables
        assert "artifactId" in variables
        assert "version" in variables

    def test_get_variables_nonexistent_template(self, server_with_temp_templates):
        """Test variable extraction for nonexistent template."""
        variables = server_with_temp_templates.get_template_variables(
            "cookiecutter", "nonexistent"
        )

        assert variables == []


class TestLangGangMCPServerSearch:
    """Tests for template search functionality."""

    def test_search_by_query(self, server_with_temp_templates):
        """Test searching templates by query string."""
        results = server_with_temp_templates.search_templates(query="test")

        assert len(results) > 0
        # All results should contain "test" in name
        for r in results:
            assert "test" in r["name"].lower()

    def test_search_by_template_type(self, server_with_temp_templates):
        """Test filtering by template type."""
        results = server_with_temp_templates.search_templates(
            template_type="cookiecutter"
        )

        for r in results:
            assert r["template_type"] == "cookiecutter"

    def test_search_with_min_score(self, server_with_temp_templates):
        """Test minimum score filtering."""
        # Low min score - should include all
        results_low = server_with_temp_templates.search_templates(
            query="test", min_score=0.1
        )

        # High min score - should include fewer
        results_high = server_with_temp_templates.search_templates(
            query="test", min_score=0.9
        )

        assert len(results_low) >= len(results_high)

    def test_search_sorted_by_relevance(self, server_with_temp_templates):
        """Test that results are sorted by relevance."""
        results = server_with_temp_templates.search_templates(query="test")

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["relevance_score"] >= results[i + 1]["relevance_score"]

    def test_search_no_criteria_returns_all(self, server_with_temp_templates):
        """Test that search without criteria returns all templates."""
        results = server_with_temp_templates.search_templates()

        templates = server_with_temp_templates.list_templates()
        total_count = sum(len(v) for v in templates.values())

        assert len(results) == total_count


# ============================================================================
# Security Tests
# ============================================================================

class TestSecurityValidation:
    """Tests for security validation functions."""

    def test_validate_path_allows_valid(self):
        """Test that valid paths are allowed."""
        _validate_path("my-template", "template_name")  # Should not raise
        _validate_path("template_v2", "template_name")  # Should not raise

    def test_validate_path_blocks_traversal(self):
        """Test that path traversal is blocked."""
        with pytest.raises(ValueError) as exc_info:
            _validate_path("../../../etc/passwd", "template_name")

        assert "must not contain" in str(exc_info.value)

    def test_validate_path_blocks_absolute(self):
        """Test that absolute paths are blocked."""
        with pytest.raises(ValueError):
            _validate_path("/etc/passwd", "template_name")

    def test_sanitize_value_allows_valid(self):
        """Test that valid property values are allowed."""
        assert _sanitize_property_value("my-project") == "my-project"
        assert _sanitize_property_value("com.example") == "com.example"
        assert _sanitize_property_value("1.0.0") == "1.0.0"

    def test_sanitize_value_blocks_injection(self):
        """Test that shell metacharacters are blocked."""
        with pytest.raises(ValueError):
            _sanitize_property_value("$(whoami)")

        with pytest.raises(ValueError):
            _sanitize_property_value("test; rm -rf /")

        with pytest.raises(ValueError):
            _sanitize_property_value("test`id`")


# ============================================================================
# Code Generation Tests
# ============================================================================

class TestCodeGeneration:
    """Tests for code generation functions."""

    def test_generate_cookiecutter_nonexistent(self, server_with_temp_templates):
        """Test error handling for nonexistent Cookiecutter template."""
        result = server_with_temp_templates.generate_from_cookiecutter(
            "nonexistent-template",
            "/tmp/output",
            {}
        )

        assert "error" in result

    def test_generate_copier_nonexistent(self, server_with_temp_templates):
        """Test error handling for nonexistent Copier template."""
        result = server_with_temp_templates.generate_from_copier(
            "nonexistent-template",
            "/tmp/output",
            {}
        )

        assert "error" in result

    def test_generate_maven_nonexistent(self, server_with_temp_templates):
        """Test error handling for nonexistent Maven archetype."""
        result = server_with_temp_templates.generate_from_maven_archetype(
            "nonexistent-archetype",
            "/tmp/output",
            {}
        )

        assert "error" in result

    def test_generate_blocks_path_traversal(self, server_with_temp_templates):
        """Test that path traversal is blocked in generation."""
        result = server_with_temp_templates.generate_from_cookiecutter(
            "../../../etc",
            "/tmp/output",
            {}
        )

        assert "error" in result
        assert "must not contain" in result["error"]
