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
- Multi-agent orchestration with six-phase execution
- Distributed resource locking and inter-agent communication
"""

__version__ = "0.3.0"

from .mcp_server import LangGangMCPServer, TemplateMetadata
from .langgraph_integration import (
    TemplateGenerationState,
    ValidationState,
    create_template_generation_graph,
    create_validation_subgraph,
    generate_project,
)
from .orchestration import (
    # Context Preservation
    KeyDecision,
    PhaseHandoffContext,
    ContextPreservingState,
    create_handoff_context,
    record_decision,
    # Enhanced Checkpointing
    CheckpointTrigger,
    EnhancedCheckpointer,
    checkpoint_before_risky,
    # Lock Manager
    LockType,
    LockManager,
    # Message Bus
    MessageType,
    ChannelType,
    MessageBus,
    # Multi-Agent Architecture
    PhaseType,
    AgentConfig,
    MultiAgentState,
    create_phase_subgraph,
    create_orchestrated_workflow,
    run_orchestrated_generation,
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
    # Orchestration - Context Preservation
    "KeyDecision",
    "PhaseHandoffContext",
    "ContextPreservingState",
    "create_handoff_context",
    "record_decision",
    # Orchestration - Checkpointing
    "CheckpointTrigger",
    "EnhancedCheckpointer",
    "checkpoint_before_risky",
    # Orchestration - Lock Manager
    "LockType",
    "LockManager",
    # Orchestration - Message Bus
    "MessageType",
    "ChannelType",
    "MessageBus",
    # Orchestration - Multi-Agent
    "PhaseType",
    "AgentConfig",
    "MultiAgentState",
    "create_phase_subgraph",
    "create_orchestrated_workflow",
    "run_orchestrated_generation",
]
