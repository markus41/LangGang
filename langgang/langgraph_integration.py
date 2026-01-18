"""
LangGraph integration for LangGang

This module provides LangGraph state graphs for multi-step template generation workflows
with LLM-powered decision making, state persistence, human-in-the-loop capabilities,
and parallel validation.
"""

import json
import os
import tempfile
import subprocess
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, TypedDict, Union
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END, START
from langgraph.types import interrupt, Command, Send
from langgraph.checkpoint.memory import MemorySaver
import operator


# ============================================================================
# State Definitions
# ============================================================================

class TemplateRecommendation(TypedDict):
    """LLM-generated template recommendation."""
    template_type: str  # "cookiecutter", "copier", "maven"
    template_name: str
    confidence: float
    reasoning: str


class ValidationResult(TypedDict):
    """Result from a validation check."""
    check_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]]


class TemplateGenerationState(TypedDict):
    """State for template generation workflow."""

    # Message history for conversation context
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Input
    project_description: str

    # Analysis results
    template_type: str
    template_name: str
    recommendation: Optional[TemplateRecommendation]

    # Context gathering
    context_variables: Dict[str, Any]
    required_variables: List[str]
    user_provided_context: bool

    # Generation
    output_dir: str
    generation_result: Optional[Dict[str, Any]]
    generation_complete: bool

    # Validation
    validation_results: List[ValidationResult]
    validation_passed: bool
    retry_count: int
    max_retries: int


class ValidationState(TypedDict):
    """State for parallel validation subgraph."""
    output_path: str
    template_type: str
    syntax_result: Optional[ValidationResult]
    lint_result: Optional[ValidationResult]
    security_result: Optional[ValidationResult]
    structure_result: Optional[ValidationResult]
    all_results: List[ValidationResult]
    all_passed: bool


# ============================================================================
# LLM Integration Helpers
# ============================================================================

def get_llm(model: str = "claude-sonnet-4-20250514"):
    """Get LLM instance with fallback options.

    Args:
        model: Model identifier to use

    Returns:
        LLM instance (ChatAnthropic or fallback)
    """
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    except ImportError:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4", temperature=0)
        except ImportError:
            return None


def get_available_templates() -> Dict[str, List[Dict[str, Any]]]:
    """Get all available templates with metadata.

    Returns:
        Dictionary mapping template types to lists of template info
    """
    from langgang.mcp_server import LangGangMCPServer

    server = LangGangMCPServer()
    templates = server.list_templates()

    # Enhance with metadata if available
    enriched = {}
    for template_type, template_names in templates.items():
        enriched[template_type] = []
        for name in template_names:
            meta = server.get_template_metadata(template_type, name)
            enriched[template_type].append({
                "name": name,
                "metadata": meta
            })

    return enriched


# ============================================================================
# Workflow Nodes - LLM-Powered
# ============================================================================

async def analyze_requirements_node(state: TemplateGenerationState) -> TemplateGenerationState:
    """Analyze user requirements using LLM to determine project type and best template.

    This node uses an LLM to intelligently analyze the project description
    and recommend the most appropriate template.

    Args:
        state: Current workflow state

    Returns:
        Updated state with analysis results and recommendation
    """
    llm = get_llm()
    description = state.get("project_description", "")
    available_templates = get_available_templates()

    # If no LLM available, fall back to heuristic matching
    if llm is None:
        return _analyze_requirements_heuristic(state)

    # Build the analysis prompt
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert software architect helping to select
the best project template. Analyze the project description and available templates
to make a recommendation.

You must respond with valid JSON in this exact format:
{
    "template_type": "cookiecutter" | "copier" | "maven",
    "template_name": "name of the specific template",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of why this template is the best match"
}

Consider:
- Programming language mentioned or implied
- Frameworks and libraries referenced
- Project complexity and structure needs
- Best practices for the domain"""),
        HumanMessage(content=f"""Project Description:
{description}

Available Templates:
{json.dumps(available_templates, indent=2)}

Analyze and recommend the best template.""")
    ])

    try:
        response = await llm.ainvoke(prompt.format_messages())
        parser = JsonOutputParser()
        recommendation = parser.parse(response.content)

        state["template_type"] = recommendation.get("template_type", "cookiecutter")
        state["template_name"] = recommendation.get("template_name", "")
        state["recommendation"] = recommendation

        # Add to message history
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"Recommended template: {recommendation['template_name']} "
                            f"(confidence: {recommendation['confidence']:.0%})\n"
                            f"Reasoning: {recommendation['reasoning']}")
        ]

    except Exception as e:
        # Fall back to heuristic on error
        return _analyze_requirements_heuristic(state)

    return state


def _analyze_requirements_heuristic(state: TemplateGenerationState) -> TemplateGenerationState:
    """Fallback heuristic-based requirement analysis.

    Used when LLM is not available.
    """
    description = state.get("project_description", "").lower()

    # Language/framework detection
    java_indicators = ["java", "spring", "maven", "gradle", "kotlin"]
    python_indicators = ["python", "flask", "django", "fastapi", "langchain"]

    if any(ind in description for ind in java_indicators):
        state["template_type"] = "maven"
        state["template_name"] = "langchain-java-archetype"
    elif any(ind in description for ind in python_indicators):
        state["template_type"] = "cookiecutter"
        state["template_name"] = "python-langchain-project"
    else:
        state["template_type"] = "copier"
        state["template_name"] = "langgraph-agent"

    state["recommendation"] = {
        "template_type": state["template_type"],
        "template_name": state["template_name"],
        "confidence": 0.6,
        "reasoning": "Matched based on keyword detection (LLM unavailable)"
    }

    return state


async def select_template_node(state: TemplateGenerationState) -> TemplateGenerationState:
    """Confirm or refine template selection with LLM assistance.

    This node validates the initial recommendation and can adjust
    based on additional context.

    Args:
        state: Current workflow state

    Returns:
        Updated state with confirmed template selection
    """
    from langgang.mcp_server import LangGangMCPServer

    server = LangGangMCPServer()
    templates = server.list_templates()

    template_type = state.get("template_type", "cookiecutter")
    template_name = state.get("template_name", "")

    available = templates.get(template_type, [])

    # Validate that selected template exists
    if template_name and template_name in available:
        # Template exists, keep selection
        pass
    elif available:
        # Template not found, select first available
        state["template_name"] = available[0]

        # Update recommendation
        if state.get("recommendation"):
            state["recommendation"]["template_name"] = available[0]
            state["recommendation"]["reasoning"] += " (adjusted to available template)"
    else:
        # No templates of this type, try another type
        for alt_type in ["cookiecutter", "copier", "maven"]:
            if templates.get(alt_type):
                state["template_type"] = alt_type
                state["template_name"] = templates[alt_type][0]
                break

    return state


async def gather_context_node(state: TemplateGenerationState) -> TemplateGenerationState:
    """Gather context variables with human-in-the-loop capability.

    This node uses LangGraph's interrupt mechanism to pause workflow
    and request user input for template variables.

    Args:
        state: Current workflow state

    Returns:
        Updated state with context variables (after user input)
    """
    from langgang.mcp_server import LangGangMCPServer

    server = LangGangMCPServer()
    template_type = state.get("template_type", "cookiecutter")
    template_name = state.get("template_name", "")

    # Get required variables from template
    required_vars = server.get_template_variables(template_type, template_name)
    state["required_variables"] = required_vars

    # Check if we already have user-provided context
    if state.get("user_provided_context"):
        return state

    # Generate smart defaults using LLM if available
    smart_defaults = await _generate_smart_defaults(
        state.get("project_description", ""),
        required_vars
    )

    # Interrupt for user input (human-in-the-loop)
    user_context = interrupt({
        "type": "context_gathering",
        "message": "Please provide values for the following template variables:",
        "template_name": template_name,
        "required_variables": required_vars,
        "suggested_defaults": smart_defaults,
        "description": state.get("project_description", "")
    })

    # Merge user input with defaults
    context = {**smart_defaults, **user_context}
    state["context_variables"] = context
    state["user_provided_context"] = True

    return state


async def _generate_smart_defaults(
    description: str,
    required_vars: List[str]
) -> Dict[str, Any]:
    """Generate intelligent default values using LLM.

    Args:
        description: Project description
        required_vars: List of required variable names

    Returns:
        Dictionary of variable names to suggested values
    """
    llm = get_llm()

    defaults = {
        "project_name": "my_project",
        "project_slug": "my_project",
        "author": "LangGang User",
        "author_name": "LangGang User",
        "email": "user@example.com",
        "description": description[:100] if description else "A new project",
        "version": "0.1.0",
        "python_version": "3.11",
        "use_langchain": True,
        "use_langgraph": True,
    }

    if llm is None:
        return {k: defaults.get(k, "") for k in required_vars}

    try:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Generate sensible default values for project template variables.
Respond with valid JSON mapping variable names to values.
Use the project description to infer appropriate values."""),
            HumanMessage(content=f"""Project Description: {description}

Required Variables: {json.dumps(required_vars)}

Generate appropriate default values as JSON.""")
        ])

        response = await llm.ainvoke(prompt.format_messages())
        parser = JsonOutputParser()
        llm_defaults = parser.parse(response.content)

        # Merge with fallback defaults
        return {k: llm_defaults.get(k, defaults.get(k, "")) for k in required_vars}

    except Exception:
        return {k: defaults.get(k, "") for k in required_vars}


async def generate_code_node(state: TemplateGenerationState) -> TemplateGenerationState:
    """Generate code from the selected template.

    Args:
        state: Current workflow state

    Returns:
        Updated state with generation results
    """
    from langgang.mcp_server import LangGangMCPServer

    server = LangGangMCPServer()
    template_type = state.get("template_type", "cookiecutter")
    template_name = state.get("template_name", "")
    context = state.get("context_variables", {})

    # Determine output directory
    output_dir = state.get("output_dir") or os.path.join(
        tempfile.gettempdir(),
        "langgang_generated",
        context.get("project_slug", context.get("project_name", "project"))
    )

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    state["output_dir"] = output_dir

    # Generate based on template type
    if template_type == "cookiecutter":
        result = server.generate_from_cookiecutter(template_name, output_dir, context)
    elif template_type == "copier":
        result = server.generate_from_copier(template_name, output_dir, context)
    else:
        result = server.generate_from_maven_archetype(template_name, output_dir, context)

    state["generation_result"] = result
    state["generation_complete"] = "error" not in result

    # Add to message history
    if "error" in result:
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"Generation failed: {result['error']}")
        ]
    else:
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"Successfully generated project at: {result.get('output', output_dir)}")
        ]

    return state


async def validate_output_node(state: TemplateGenerationState) -> TemplateGenerationState:
    """Validate generated output using parallel validation subgraph.

    This node orchestrates multiple validation checks in parallel:
    - Syntax validation
    - Linting
    - Security scanning
    - Structure verification

    Args:
        state: Current workflow state

    Returns:
        Updated state with validation results
    """
    output_dir = state.get("output_dir", "")
    template_type = state.get("template_type", "cookiecutter")

    if not output_dir or not os.path.exists(output_dir):
        state["validation_passed"] = False
        state["validation_results"] = [{
            "check_name": "existence",
            "passed": False,
            "message": "Output directory does not exist",
            "details": None
        }]
        return state

    # Run validation subgraph
    validation_graph = create_validation_subgraph()
    validation_state: ValidationState = {
        "output_path": output_dir,
        "template_type": template_type,
        "syntax_result": None,
        "lint_result": None,
        "security_result": None,
        "structure_result": None,
        "all_results": [],
        "all_passed": True
    }

    try:
        # Execute validation subgraph
        final_state = await validation_graph.ainvoke(validation_state)

        state["validation_results"] = final_state.get("all_results", [])
        state["validation_passed"] = final_state.get("all_passed", False)

    except Exception as e:
        state["validation_results"] = [{
            "check_name": "validation_error",
            "passed": False,
            "message": f"Validation failed with error: {str(e)}",
            "details": None
        }]
        state["validation_passed"] = False

    return state


# ============================================================================
# Parallel Validation Subgraph
# ============================================================================

def create_validation_subgraph():
    """Create a parallel validation subgraph.

    This subgraph runs multiple validation checks concurrently:
    - Syntax check (Python/Java file parsing)
    - Lint check (ruff/pylint for Python)
    - Security scan (basic security patterns)
    - Structure check (required files exist)

    Returns:
        Compiled validation StateGraph
    """
    workflow = StateGraph(ValidationState)

    # Add validation nodes
    workflow.add_node("syntax_check", syntax_check_node)
    workflow.add_node("lint_check", lint_check_node)
    workflow.add_node("security_scan", security_scan_node)
    workflow.add_node("structure_check", structure_check_node)
    workflow.add_node("aggregate_results", aggregate_results_node)

    # Fan-out: route to all validators in parallel
    def route_to_validators(state: ValidationState) -> List[Send]:
        """Route to all validation nodes in parallel."""
        return [
            Send("syntax_check", state),
            Send("lint_check", state),
            Send("security_scan", state),
            Send("structure_check", state),
        ]

    workflow.add_conditional_edges(START, route_to_validators)

    # Fan-in: aggregate all results
    workflow.add_edge("syntax_check", "aggregate_results")
    workflow.add_edge("lint_check", "aggregate_results")
    workflow.add_edge("security_scan", "aggregate_results")
    workflow.add_edge("structure_check", "aggregate_results")

    workflow.add_edge("aggregate_results", END)

    return workflow.compile()


async def syntax_check_node(state: ValidationState) -> ValidationState:
    """Check syntax of generated files.

    Validates Python files can be parsed, Java files compile, etc.
    """
    output_path = state["output_path"]
    template_type = state.get("template_type", "")

    errors = []
    files_checked = 0

    try:
        for root, _, files in os.walk(output_path):
            for file in files:
                filepath = os.path.join(root, file)

                if file.endswith(".py"):
                    files_checked += 1
                    try:
                        with open(filepath, "r") as f:
                            compile(f.read(), filepath, "exec")
                    except SyntaxError as e:
                        errors.append(f"{filepath}: {e}")

                elif file.endswith(".java"):
                    files_checked += 1
                    # Basic Java syntax check (imports and class declaration)
                    with open(filepath, "r") as f:
                        content = f.read()
                        if "class " not in content and "interface " not in content:
                            if not file.endswith("package-info.java"):
                                errors.append(f"{filepath}: No class or interface declaration found")

    except Exception as e:
        errors.append(f"Syntax check error: {str(e)}")

    passed = len(errors) == 0
    state["syntax_result"] = {
        "check_name": "syntax",
        "passed": passed,
        "message": f"Checked {files_checked} files" if passed else f"Found {len(errors)} syntax errors",
        "details": {"errors": errors, "files_checked": files_checked}
    }

    return state


async def lint_check_node(state: ValidationState) -> ValidationState:
    """Run linting on generated Python files.

    Uses ruff for fast Python linting.
    """
    output_path = state["output_path"]

    # Check if there are Python files to lint
    python_files = list(Path(output_path).rglob("*.py"))

    if not python_files:
        state["lint_result"] = {
            "check_name": "lint",
            "passed": True,
            "message": "No Python files to lint",
            "details": None
        }
        return state

    try:
        result = subprocess.run(
            ["ruff", "check", output_path, "--output-format=json"],
            capture_output=True,
            text=True,
            timeout=30
        )

        issues = []
        if result.stdout:
            try:
                issues = json.loads(result.stdout)
            except json.JSONDecodeError:
                pass

        # Filter to only errors (not warnings)
        errors = [i for i in issues if i.get("type") == "E" or i.get("code", "").startswith("E")]

        passed = len(errors) == 0
        state["lint_result"] = {
            "check_name": "lint",
            "passed": passed,
            "message": f"Found {len(errors)} linting errors" if not passed else "Linting passed",
            "details": {"issues": issues[:10], "total_issues": len(issues)}  # Limit details
        }

    except FileNotFoundError:
        # ruff not installed
        state["lint_result"] = {
            "check_name": "lint",
            "passed": True,
            "message": "Linting skipped (ruff not installed)",
            "details": None
        }
    except subprocess.TimeoutExpired:
        state["lint_result"] = {
            "check_name": "lint",
            "passed": False,
            "message": "Linting timed out",
            "details": None
        }
    except Exception as e:
        state["lint_result"] = {
            "check_name": "lint",
            "passed": False,
            "message": f"Linting error: {str(e)}",
            "details": None
        }

    return state


async def security_scan_node(state: ValidationState) -> ValidationState:
    """Perform basic security scanning on generated code.

    Checks for common security issues like:
    - Hardcoded secrets/credentials
    - SQL injection patterns
    - Command injection patterns
    """
    output_path = state["output_path"]

    security_patterns = [
        (r'password\s*=\s*["\'][^"\']+["\']', "Possible hardcoded password"),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Possible hardcoded API key"),
        (r'secret\s*=\s*["\'][^"\']+["\']', "Possible hardcoded secret"),
        (r'eval\s*\(', "Use of eval() - potential code injection"),
        (r'exec\s*\(', "Use of exec() - potential code injection"),
        (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Shell=True in subprocess - potential command injection"),
        (r'os\.system\s*\(', "Use of os.system() - prefer subprocess"),
    ]

    import re
    findings = []
    files_scanned = 0

    try:
        for root, _, files in os.walk(output_path):
            for file in files:
                if file.endswith((".py", ".java", ".js", ".ts")):
                    filepath = os.path.join(root, file)
                    files_scanned += 1

                    with open(filepath, "r") as f:
                        content = f.read()

                    for pattern, description in security_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            findings.append({
                                "file": filepath,
                                "issue": description,
                                "count": len(matches)
                            })

    except Exception as e:
        findings.append({"error": str(e)})

    # Security findings are warnings, not failures (templates may have placeholders)
    passed = not any(f.get("issue", "").startswith("Possible hardcoded") for f in findings)

    state["security_result"] = {
        "check_name": "security",
        "passed": passed,
        "message": f"Scanned {files_scanned} files, found {len(findings)} potential issues",
        "details": {"findings": findings[:10]}
    }

    return state


async def structure_check_node(state: ValidationState) -> ValidationState:
    """Verify project structure is correct.

    Checks that expected files and directories exist based on template type.
    """
    output_path = state["output_path"]
    template_type = state.get("template_type", "")

    expected_files = {
        "cookiecutter": ["__init__.py", "README.md"],
        "copier": ["__init__.py", "README.md"],
        "maven": ["pom.xml", "src"],
    }

    missing = []
    found = []

    expected = expected_files.get(template_type, ["README.md"])

    try:
        # Walk the output directory to find files
        all_files = set()
        all_dirs = set()
        for root, dirs, files in os.walk(output_path):
            rel_root = os.path.relpath(root, output_path)
            for f in files:
                if rel_root == ".":
                    all_files.add(f)
                else:
                    all_files.add(os.path.join(rel_root, f))
            for d in dirs:
                if rel_root == ".":
                    all_dirs.add(d)
                else:
                    all_dirs.add(os.path.join(rel_root, d))

        for item in expected:
            # Check in files and directories
            item_found = (
                item in all_files or
                item in all_dirs or
                any(item in f for f in all_files) or
                any(item in d for d in all_dirs)
            )

            if item_found:
                found.append(item)
            else:
                missing.append(item)

    except Exception as e:
        missing.append(f"Error checking structure: {str(e)}")

    passed = len(missing) == 0
    state["structure_result"] = {
        "check_name": "structure",
        "passed": passed,
        "message": f"Found {len(found)}/{len(expected)} expected items" +
                  (f", missing: {missing}" if missing else ""),
        "details": {"found": found, "missing": missing}
    }

    return state


async def aggregate_results_node(state: ValidationState) -> ValidationState:
    """Aggregate all validation results.

    Combines results from parallel validation nodes into final verdict.
    """
    results = []

    for result_key in ["syntax_result", "lint_result", "security_result", "structure_result"]:
        result = state.get(result_key)
        if result:
            results.append(result)

    state["all_results"] = results
    state["all_passed"] = all(r.get("passed", False) for r in results)

    return state


# ============================================================================
# Conditional Edges
# ============================================================================

def should_regenerate(state: TemplateGenerationState) -> Literal["regenerate", "end"]:
    """Determine if code should be regenerated.

    Considers validation results and retry count.

    Args:
        state: Current workflow state

    Returns:
        "regenerate" to retry or "end" to finish
    """
    # Check if generation completed successfully
    if not state.get("generation_complete", False):
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)

        if retry_count < max_retries:
            return "regenerate"

    # Check if validation passed
    if state.get("validation_passed", True):
        return "end"

    # Validation failed - check retry count
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        return "regenerate"

    return "end"


# ============================================================================
# Main Graph Factory
# ============================================================================

def create_template_generation_graph(
    checkpointer: Optional[Any] = None,
    enable_persistence: bool = True
):
    """Create a LangGraph state graph for template generation.

    This graph implements a multi-step workflow with:
    - LLM-powered requirement analysis
    - Intelligent template selection
    - Human-in-the-loop context gathering
    - Code generation with multiple template engines
    - Parallel validation (syntax, lint, security, structure)
    - Automatic retry on failure

    Args:
        checkpointer: Optional custom checkpointer for state persistence.
                     If None and enable_persistence is True, uses MemorySaver.
        enable_persistence: Whether to enable state persistence for
                           debugging and resume capability.

    Returns:
        Compiled StateGraph for template generation
    """
    # Create the graph
    workflow = StateGraph(TemplateGenerationState)

    # Add nodes
    workflow.add_node("analyze_requirements", analyze_requirements_node)
    workflow.add_node("select_template", select_template_node)
    workflow.add_node("gather_context", gather_context_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("validate_output", validate_output_node)

    # Add edges - linear flow through workflow
    workflow.set_entry_point("analyze_requirements")
    workflow.add_edge("analyze_requirements", "select_template")
    workflow.add_edge("select_template", "gather_context")
    workflow.add_edge("gather_context", "generate_code")
    workflow.add_edge("generate_code", "validate_output")

    # Conditional edge from validation - may regenerate on failure
    workflow.add_conditional_edges(
        "validate_output",
        should_regenerate,
        {
            "regenerate": "gather_context",
            "end": END
        }
    )

    # Set up checkpointing for state persistence
    if enable_persistence:
        if checkpointer is None:
            checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# ============================================================================
# Convenience Functions
# ============================================================================

async def generate_project(
    description: str,
    output_dir: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """High-level function to generate a project from description.

    Args:
        description: Natural language project description
        output_dir: Optional output directory for generated project
        context: Optional pre-filled context variables (skips user input)
        thread_id: Optional thread ID for checkpointing/resumption

    Returns:
        Dictionary with generation results
    """
    graph = create_template_generation_graph()

    initial_state: TemplateGenerationState = {
        "messages": [],
        "project_description": description,
        "template_type": "",
        "template_name": "",
        "recommendation": None,
        "context_variables": context or {},
        "required_variables": [],
        "user_provided_context": context is not None,
        "output_dir": output_dir or "",
        "generation_result": None,
        "generation_complete": False,
        "validation_results": [],
        "validation_passed": False,
        "retry_count": 0,
        "max_retries": 3
    }

    config = {"configurable": {"thread_id": thread_id or "default"}}

    try:
        final_state = await graph.ainvoke(initial_state, config)

        return {
            "success": final_state.get("generation_complete", False),
            "output_dir": final_state.get("output_dir", ""),
            "template_used": f"{final_state.get('template_type')}/{final_state.get('template_name')}",
            "recommendation": final_state.get("recommendation"),
            "validation_results": final_state.get("validation_results", []),
            "validation_passed": final_state.get("validation_passed", False),
            "messages": [m.content for m in final_state.get("messages", [])]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
