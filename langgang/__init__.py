"""
LangGang - AI-powered code templating framework

This package provides integration between LangChain, LangGraph, Cookiecutter,
Copier, and Maven archetypes for intelligent code scaffolding.
"""

__version__ = "0.1.0"

from .mcp_server import LangGangMCPServer

__all__ = ["LangGangMCPServer"]
