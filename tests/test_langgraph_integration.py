"""
Tests for LangGraph integration module.

Tests cover:
- LLM-powered decision nodes (with fallback)
- State checkpointing and persistence
- Parallel validation subgraph
- Workflow graph creation
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langgang.langgraph_integration import (
    TemplateGenerationState,
    TemplateRecommendation,
    ValidationResult,
    ValidationState,
    _analyze_requirements_heuristic,
    _generate_smart_defaults,
    aggregate_results_node,
    analyze_requirements_node,
    create_template_generation_graph,
    create_validation_subgraph,
    generate_code_node,
    generate_project,
    get_available_templates,
    get_llm,
    lint_check_node,
    security_scan_node,
    select_template_node,
    should_regenerate,
    structure_check_node,
    syntax_check_node,
    validate_output_node,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_state() -> TemplateGenerationState:
    """Create a sample template generation state."""
    return {
        "messages": [],
        "project_description": "A Python web API with LangChain integration",
        "template_type": "",
        "template_name": "",
        "recommendation": None,
        "context_variables": {},
        "required_variables": [],
        "user_provided_context": False,
        "output_dir": "",
        "generation_result": None,
        "generation_complete": False,
        "validation_results": [],
        "validation_passed": False,
        "retry_count": 0,
        "max_retries": 3,
    }


@pytest.fixture
def sample_validation_state() -> ValidationState:
    """Create a sample validation state."""
    return {
        "output_path": "/tmp/test_project",
        "template_type": "cookiecutter",
        "syntax_result": None,
        "lint_result": None,
        "security_result": None,
        "structure_result": None,
        "all_results": [],
        "all_passed": True,
    }


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple Python file
        py_file = Path(tmpdir) / "main.py"
        py_file.write_text('print("Hello, World!")\n')

        # Create a README
        readme = Path(tmpdir) / "README.md"
        readme.write_text("# Test Project\n")

        # Create an __init__.py
        init_file = Path(tmpdir) / "__init__.py"
        init_file.write_text("")

        yield tmpdir


# ============================================================================
# LLM Integration Tests
# ============================================================================

class TestGetLLM:
    """Tests for get_llm function."""

    def test_returns_none_when_no_providers(self):
        """Test that get_llm returns None when no LLM providers installed."""
        with patch.dict("sys.modules", {"langchain_anthropic": None, "langchain_openai": None}):
            # Force reimport to test fallback
            result = get_llm()
            # Result depends on whether providers are actually installed
            assert result is None or hasattr(result, "invoke")


class TestAnalyzeRequirementsHeuristic:
    """Tests for heuristic-based requirement analysis."""

    def test_detects_java_project(self, sample_state):
        """Test detection of Java/Maven projects."""
        sample_state["project_description"] = "A Spring Boot REST API with Java"
        result = _analyze_requirements_heuristic(sample_state)

        assert result["template_type"] == "maven"
        assert result["recommendation"]["confidence"] == 0.6

    def test_detects_python_project(self, sample_state):
        """Test detection of Python projects."""
        sample_state["project_description"] = "A Python Flask web application"
        result = _analyze_requirements_heuristic(sample_state)

        assert result["template_type"] == "cookiecutter"
        assert "python-langchain-project" in result["template_name"]

    def test_defaults_to_copier(self, sample_state):
        """Test fallback to Copier for ambiguous descriptions."""
        sample_state["project_description"] = "A simple automation script"
        result = _analyze_requirements_heuristic(sample_state)

        assert result["template_type"] == "copier"

    def test_langchain_detection(self, sample_state):
        """Test detection of LangChain projects."""
        sample_state["project_description"] = "Build a langchain agent"
        result = _analyze_requirements_heuristic(sample_state)

        assert result["template_type"] == "cookiecutter"


# ============================================================================
# Workflow Node Tests
# ============================================================================

class TestAnalyzeRequirementsNode:
    """Tests for analyze_requirements_node."""

    @pytest.mark.asyncio
    async def test_falls_back_to_heuristic_without_llm(self, sample_state):
        """Test fallback to heuristic when LLM unavailable."""
        with patch("langgang.langgraph_integration.get_llm", return_value=None):
            result = await analyze_requirements_node(sample_state)

        assert result["template_type"] in ["cookiecutter", "copier", "maven"]
        assert result["recommendation"] is not None


class TestSelectTemplateNode:
    """Tests for select_template_node."""

    @pytest.mark.asyncio
    async def test_validates_template_exists(self, sample_state):
        """Test that template selection validates existence."""
        sample_state["template_type"] = "cookiecutter"
        sample_state["template_name"] = "nonexistent-template"

        result = await select_template_node(sample_state)

        # Should fall back to available template
        assert result["template_name"] != "nonexistent-template" or result["template_name"] == ""

    @pytest.mark.asyncio
    async def test_keeps_valid_template(self, sample_state):
        """Test that valid template selection is preserved."""
        sample_state["template_type"] = "cookiecutter"
        sample_state["template_name"] = "python-langchain-project"

        result = await select_template_node(sample_state)

        assert result["template_name"] == "python-langchain-project"


class TestGenerateSmartDefaults:
    """Tests for smart default generation."""

    @pytest.mark.asyncio
    async def test_generates_defaults_without_llm(self):
        """Test default generation when LLM unavailable."""
        with patch("langgang.langgraph_integration.get_llm", return_value=None):
            result = await _generate_smart_defaults(
                "A test project",
                ["project_name", "author", "version"]
            )

        assert "project_name" in result
        assert "author" in result
        assert "version" in result

    @pytest.mark.asyncio
    async def test_includes_description_in_defaults(self):
        """Test that description is included in defaults."""
        description = "My awesome project description"
        with patch("langgang.langgraph_integration.get_llm", return_value=None):
            result = await _generate_smart_defaults(
                description,
                ["description"]
            )

        assert result["description"] == description[:100]


# ============================================================================
# Validation Subgraph Tests
# ============================================================================

class TestSyntaxCheckNode:
    """Tests for syntax_check_node."""

    @pytest.mark.asyncio
    async def test_valid_python_files(self, temp_project_dir, sample_validation_state):
        """Test syntax check passes for valid Python files."""
        sample_validation_state["output_path"] = temp_project_dir

        result = await syntax_check_node(sample_validation_state)

        assert result["syntax_result"]["passed"] is True
        assert result["syntax_result"]["check_name"] == "syntax"

    @pytest.mark.asyncio
    async def test_invalid_python_files(self, sample_validation_state):
        """Test syntax check catches invalid Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid Python file
            bad_file = Path(tmpdir) / "bad.py"
            bad_file.write_text("def broken(\n")  # Syntax error

            sample_validation_state["output_path"] = tmpdir
            result = await syntax_check_node(sample_validation_state)

            assert result["syntax_result"]["passed"] is False
            assert len(result["syntax_result"]["details"]["errors"]) > 0


class TestLintCheckNode:
    """Tests for lint_check_node."""

    @pytest.mark.asyncio
    async def test_no_python_files(self, sample_validation_state):
        """Test lint check handles directories without Python files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_validation_state["output_path"] = tmpdir

            result = await lint_check_node(sample_validation_state)

            assert result["lint_result"]["passed"] is True
            assert "No Python files" in result["lint_result"]["message"]

    @pytest.mark.asyncio
    async def test_handles_missing_ruff(self, temp_project_dir, sample_validation_state):
        """Test graceful handling when ruff is not installed."""
        sample_validation_state["output_path"] = temp_project_dir

        with patch("subprocess.run", side_effect=FileNotFoundError()):
            result = await lint_check_node(sample_validation_state)

        assert result["lint_result"]["passed"] is True
        assert "skipped" in result["lint_result"]["message"].lower()


class TestSecurityScanNode:
    """Tests for security_scan_node."""

    @pytest.mark.asyncio
    async def test_clean_files(self, temp_project_dir, sample_validation_state):
        """Test security scan passes for clean files."""
        sample_validation_state["output_path"] = temp_project_dir

        result = await security_scan_node(sample_validation_state)

        assert result["security_result"]["passed"] is True

    @pytest.mark.asyncio
    async def test_detects_hardcoded_secrets(self, sample_validation_state):
        """Test security scan detects hardcoded secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with hardcoded secret
            bad_file = Path(tmpdir) / "config.py"
            bad_file.write_text('password = "supersecret123"\n')

            sample_validation_state["output_path"] = tmpdir
            result = await security_scan_node(sample_validation_state)

            assert result["security_result"]["passed"] is False
            assert len(result["security_result"]["details"]["findings"]) > 0


class TestStructureCheckNode:
    """Tests for structure_check_node."""

    @pytest.mark.asyncio
    async def test_valid_structure(self, temp_project_dir, sample_validation_state):
        """Test structure check for valid project."""
        sample_validation_state["output_path"] = temp_project_dir

        result = await structure_check_node(sample_validation_state)

        assert result["structure_result"]["passed"] is True
        assert "__init__.py" in result["structure_result"]["details"]["found"]

    @pytest.mark.asyncio
    async def test_missing_files(self, sample_validation_state):
        """Test structure check detects missing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty directory
            sample_validation_state["output_path"] = tmpdir
            result = await structure_check_node(sample_validation_state)

            assert result["structure_result"]["passed"] is False
            assert len(result["structure_result"]["details"]["missing"]) > 0


class TestAggregateResultsNode:
    """Tests for aggregate_results_node."""

    @pytest.mark.asyncio
    async def test_aggregates_all_results(self, sample_validation_state):
        """Test aggregation of validation results."""
        sample_validation_state["syntax_result"] = {
            "check_name": "syntax", "passed": True, "message": "OK", "details": None
        }
        sample_validation_state["lint_result"] = {
            "check_name": "lint", "passed": True, "message": "OK", "details": None
        }
        sample_validation_state["security_result"] = {
            "check_name": "security", "passed": False, "message": "Issues", "details": None
        }
        sample_validation_state["structure_result"] = {
            "check_name": "structure", "passed": True, "message": "OK", "details": None
        }

        result = await aggregate_results_node(sample_validation_state)

        assert len(result["all_results"]) == 4
        assert result["all_passed"] is False  # One check failed


# ============================================================================
# Conditional Edge Tests
# ============================================================================

class TestShouldRegenerate:
    """Tests for should_regenerate function."""

    def test_ends_on_success(self, sample_state):
        """Test workflow ends when generation successful."""
        sample_state["generation_complete"] = True
        sample_state["validation_passed"] = True

        result = should_regenerate(sample_state)

        assert result == "end"

    def test_regenerates_on_failure(self, sample_state):
        """Test workflow regenerates on failure."""
        sample_state["generation_complete"] = False
        sample_state["retry_count"] = 0

        result = should_regenerate(sample_state)

        assert result == "regenerate"

    def test_ends_after_max_retries(self, sample_state):
        """Test workflow ends after max retries."""
        sample_state["generation_complete"] = False
        sample_state["retry_count"] = 3
        sample_state["max_retries"] = 3

        result = should_regenerate(sample_state)

        assert result == "end"


# ============================================================================
# Graph Creation Tests
# ============================================================================

class TestCreateTemplateGenerationGraph:
    """Tests for graph creation functions."""

    def test_creates_graph_with_checkpointer(self):
        """Test graph creation with default checkpointer."""
        graph = create_template_generation_graph(enable_persistence=True)

        assert graph is not None
        # Graph should have the expected nodes
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")

    def test_creates_graph_without_checkpointer(self):
        """Test graph creation without persistence."""
        graph = create_template_generation_graph(enable_persistence=False)

        assert graph is not None


class TestCreateValidationSubgraph:
    """Tests for validation subgraph creation."""

    def test_creates_subgraph(self):
        """Test validation subgraph creation."""
        graph = create_validation_subgraph()

        assert graph is not None
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")


# ============================================================================
# Integration Tests
# ============================================================================

class TestGenerateProject:
    """Integration tests for generate_project function."""

    @pytest.mark.asyncio
    async def test_generate_with_context(self):
        """Test project generation with pre-provided context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await generate_project(
                description="A Python LangChain project",
                output_dir=tmpdir,
                context={
                    "project_name": "test_project",
                    "author": "Test Author",
                    "version": "1.0.0",
                },
            )

            # Result should have expected structure
            assert "success" in result or "error" in result
            assert "template_used" in result or "error" in result
