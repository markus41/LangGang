"""
LangGraph integration for LangGang

This module provides LangGraph state graphs for multi-step template generation workflows.
"""

import tempfile
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
import operator


class TemplateGenerationState(TypedDict):
    """State for template generation workflow."""
    
    messages: Annotated[Sequence[BaseMessage], operator.add]
    project_description: str
    template_type: str
    template_name: str
    context_variables: dict
    generation_complete: bool


def create_template_generation_graph():
    """Create a LangGraph state graph for template generation.
    
    This graph implements a multi-step workflow:
    1. Analyze requirements
    2. Select template
    3. Gather context
    4. Generate code
    5. Validate output
    
    Returns:
        Compiled StateGraph for template generation
    """
    from langgang.langchain_integration import get_langgang_tools
    
    # Create the graph
    workflow = StateGraph(TemplateGenerationState)
    
    # Add nodes
    workflow.add_node("analyze_requirements", analyze_requirements_node)
    workflow.add_node("select_template", select_template_node)
    workflow.add_node("gather_context", gather_context_node)
    workflow.add_node("generate_code", generate_code_node)
    workflow.add_node("validate_output", validate_output_node)
    
    # Add edges
    workflow.set_entry_point("analyze_requirements")
    workflow.add_edge("analyze_requirements", "select_template")
    workflow.add_edge("select_template", "gather_context")
    workflow.add_edge("gather_context", "generate_code")
    workflow.add_edge("generate_code", "validate_output")
    
    # Conditional edge from validation
    workflow.add_conditional_edges(
        "validate_output",
        should_regenerate,
        {
            "regenerate": "gather_context",
            "end": END
        }
    )
    
    return workflow.compile()


def analyze_requirements_node(state: TemplateGenerationState) -> TemplateGenerationState:
    """Analyze user requirements to determine project type.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with analysis results
    """
    # In a real implementation, this would use an LLM to analyze requirements
    # For now, we'll use simple heuristics
    description = state.get("project_description", "").lower()
    
    if "java" in description or "spring" in description or "maven" in description:
        state["template_type"] = "maven"
    elif "python" in description or "flask" in description or "django" in description:
        state["template_type"] = "cookiecutter"
    else:
        state["template_type"] = "copier"
    
    return state


def select_template_node(state: TemplateGenerationState) -> TemplateGenerationState:
    """Select the appropriate template based on requirements.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with template selection
    """
    from langgang.mcp_server import LangGangMCPServer
    
    server = LangGangMCPServer()
    templates = server.list_templates()
    
    template_type = state.get("template_type", "cookiecutter")
    available = templates.get(template_type, [])
    
    if available:
        # Select the first available template
        # In a real implementation, this would use LLM to choose the best match
        state["template_name"] = available[0]
    else:
        state["template_name"] = "default"
    
    return state


def gather_context_node(state: TemplateGenerationState) -> TemplateGenerationState:
    """Gather context variables needed for template generation.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with context variables
    """
    # In a real implementation, this would interact with the user or use LLM
    # to gather necessary context variables
    state["context_variables"] = state.get("context_variables", {
        "project_name": "my_project",
        "author": "LangGang User",
        "version": "0.1.0"
    })
    
    return state


def generate_code_node(state: TemplateGenerationState) -> TemplateGenerationState:
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
    
    # Use cross-platform temporary directory
    output_dir = tempfile.gettempdir() + "/generated_projects"
    
    if template_type == "cookiecutter":
        result = server.generate_from_cookiecutter(template_name, output_dir, context)
    elif template_type == "copier":
        result = server.generate_from_copier(template_name, output_dir, context)
    else:
        result = server.generate_from_maven_archetype(template_name, output_dir, context)
    
    state["generation_complete"] = "error" not in result
    
    return state


def validate_output_node(state: TemplateGenerationState) -> TemplateGenerationState:
    """Validate the generated output.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with validation results
    """
    # In a real implementation, this would validate the generated code
    # For now, we'll just mark it as complete
    state["generation_complete"] = True
    
    return state


def should_regenerate(state: TemplateGenerationState) -> str:
    """Determine if code should be regenerated.
    
    Args:
        state: Current workflow state
        
    Returns:
        "regenerate" or "end"
    """
    if state.get("generation_complete", False):
        return "end"
    else:
        return "regenerate"
