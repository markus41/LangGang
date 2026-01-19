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
- Circuit breaker and health monitoring
- Dynamic agent scaling
- Distributed tracing
- Semantic caching for LLM calls
- Priority-based task scheduling
- Event sourcing for state changes
- Workflow versioning and migration
- Retry policies with exponential backoff
- Workflow composition and templates
- Multi-tenancy and isolation
"""

__version__ = "0.4.0"

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
from .orchestration_v2 import (
    # Circuit Breaker
    CircuitState,
    CircuitBreaker,
    AgentHealthMonitor,
    # Dynamic Scaling
    ScalingConfig,
    AgentMetrics,
    DynamicScaler,
    # Distributed Tracing
    Span,
    Tracer,
    # Semantic Cache
    CacheEntry,
    SemanticCache,
    # Priority Scheduling
    Priority,
    ScheduledTask,
    PriorityScheduler,
    # Event Sourcing
    Event,
    WorkflowStartedEvent,
    PhaseStartedEvent,
    PhaseCompletedEvent,
    AgentSpawnedEvent,
    AgentCompletedEvent,
    DecisionMadeEvent,
    FileGeneratedEvent,
    ValidationResultEvent,
    ErrorOccurredEvent,
    WorkflowCompletedEvent,
    EventStore,
    StateProjector,
    WorkflowProjector,
    # Version Migration
    Version,
    Migration,
    StateMigrator,
    create_default_migrator,
    # Retry Policies
    RetryPolicy,
    RetryError,
    with_retry,
    RetryableOperation,
    # Workflow Composition
    WorkflowStep,
    ParallelSteps,
    SequentialSteps,
    ConditionalStep,
    LoopStep,
    step,
    parallel,
    sequence,
    when,
    loop,
    noop,
    WorkflowTemplate,
    ComposedWorkflow,
    WorkflowRegistry,
    # Multi-Tenancy
    current_tenant,
    TenantConfig,
    TenantContext,
    TenantUsage,
    TenantManager,
    IsolatedResource,
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
    # Orchestration v1 - Context Preservation
    "KeyDecision",
    "PhaseHandoffContext",
    "ContextPreservingState",
    "create_handoff_context",
    "record_decision",
    # Orchestration v1 - Checkpointing
    "CheckpointTrigger",
    "EnhancedCheckpointer",
    "checkpoint_before_risky",
    # Orchestration v1 - Lock Manager
    "LockType",
    "LockManager",
    # Orchestration v1 - Message Bus
    "MessageType",
    "ChannelType",
    "MessageBus",
    # Orchestration v1 - Multi-Agent
    "PhaseType",
    "AgentConfig",
    "MultiAgentState",
    "create_phase_subgraph",
    "create_orchestrated_workflow",
    "run_orchestrated_generation",
    # Orchestration v2 - Circuit Breaker
    "CircuitState",
    "CircuitBreaker",
    "AgentHealthMonitor",
    # Orchestration v2 - Dynamic Scaling
    "ScalingConfig",
    "AgentMetrics",
    "DynamicScaler",
    # Orchestration v2 - Distributed Tracing
    "Span",
    "Tracer",
    # Orchestration v2 - Semantic Cache
    "CacheEntry",
    "SemanticCache",
    # Orchestration v2 - Priority Scheduling
    "Priority",
    "ScheduledTask",
    "PriorityScheduler",
    # Orchestration v2 - Event Sourcing
    "Event",
    "WorkflowStartedEvent",
    "PhaseStartedEvent",
    "PhaseCompletedEvent",
    "AgentSpawnedEvent",
    "AgentCompletedEvent",
    "DecisionMadeEvent",
    "FileGeneratedEvent",
    "ValidationResultEvent",
    "ErrorOccurredEvent",
    "WorkflowCompletedEvent",
    "EventStore",
    "StateProjector",
    "WorkflowProjector",
    # Orchestration v2 - Version Migration
    "Version",
    "Migration",
    "StateMigrator",
    "create_default_migrator",
    # Orchestration v2 - Retry Policies
    "RetryPolicy",
    "RetryError",
    "with_retry",
    "RetryableOperation",
    # Orchestration v2 - Workflow Composition
    "WorkflowStep",
    "ParallelSteps",
    "SequentialSteps",
    "ConditionalStep",
    "LoopStep",
    "step",
    "parallel",
    "sequence",
    "when",
    "loop",
    "noop",
    "WorkflowTemplate",
    "ComposedWorkflow",
    "WorkflowRegistry",
    # Orchestration v2 - Multi-Tenancy
    "current_tenant",
    "TenantConfig",
    "TenantContext",
    "TenantUsage",
    "TenantManager",
    "IsolatedResource",
]
