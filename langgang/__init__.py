"""
LangGang - AI-powered code templating framework

This package provides integration between LangChain, LangGraph, Cookiecutter,
Copier, and Maven archetypes for intelligent code scaffolding.

Features:
- LLM-powered template selection and recommendation
- Multi-engine code generation (Cookiecutter, Copier, Maven)
- Rich template metadata and discovery
- Human-in-the-loop context gathering
- Parallel validation of generated code
- State persistence and checkpointing
"""

__version__ = "0.2.0"

from .mcp_server import LangGangMCPServer, TemplateMetadata
from .langgraph_integration import (
    TemplateGenerationState,
    ValidationState,
    create_template_generation_graph,
    create_validation_subgraph,
    generate_project,
)

__all__ = [
    # MCP Server
    "LangGangMCPServer",
    "TemplateMetadata",
    # LangGraph Integration
    "TemplateGenerationState",
    "ValidationState",
    "create_template_generation_graph",
    "create_validation_subgraph",
    "generate_project",
]
