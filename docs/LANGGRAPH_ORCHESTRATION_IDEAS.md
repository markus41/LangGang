# LangGraph Orchestration Ideas

> Inspired by the [Claude Orchestration Protocol v4.0.0](https://github.com/Lobbi-Docs/claude/blob/main/.claude/orchestration/PROTOCOL.md)

This document describes 5 high-impact enhancements to the LangGang LangGraph implementation, bringing advanced multi-agent orchestration patterns to the template generation workflow.

## Implementation Status: ✅ COMPLETE

All 5 orchestration ideas have been implemented in `langgang/orchestration.py`:

| # | Feature | Status | Module |
|---|---------|--------|--------|
| 1 | Six-Phase Multi-Agent Architecture | ✅ Implemented | `create_orchestrated_workflow()` |
| 2 | Distributed Lock Manager | ✅ Implemented | `LockManager` |
| 3 | Inter-Agent Communication Channels | ✅ Implemented | `MessageBus` |
| 4 | Advanced Checkpointing with Recovery | ✅ Implemented | `EnhancedCheckpointer` |
| 5 | Context Preservation Protocol | ✅ Implemented | `PhaseHandoffContext` |

### Quick Start

```python
from langgang import (
    # Multi-Agent Workflow
    run_orchestrated_generation,
    create_orchestrated_workflow,
    PhaseType,

    # Lock Manager
    LockManager,
    LockType,

    # Message Bus
    MessageBus,
    MessageType,
    ChannelType,

    # Checkpointing
    EnhancedCheckpointer,
    CheckpointTrigger,
    checkpoint_before_risky,

    # Context Preservation
    create_handoff_context,
    record_decision,
)

# Run the full orchestrated workflow
result = await run_orchestrated_generation(
    description="A Python Flask API with LangChain",
    output_dir="/tmp/my_project",
    context={"author": "Developer"},
)
```

---

## 1. Six-Phase Multi-Agent Architecture

### Current State
The existing workflow uses a linear 5-step single-agent flow:
```
analyze → select → gather_context → generate → validate
```

### Proposed Enhancement
Implement the protocol's **six-phase progression** with specialized agent roles:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PHASE-BASED ORCHESTRATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ EXPLORE  │ → │   PLAN   │ → │   CODE   │ → │   TEST   │ → │   FIX    │  │
│  │          │   │          │   │          │   │          │   │    ↓     │  │
│  │Researcher│   │ Planner  │   │ Coder    │   │ Tester   │   │Debugger  │  │
│  │Analyzer  │   │Architect │   │ Template │   │ Security │   │    ↓     │  │
│  │          │   │          │   │ Expert   │   │ E2E      │   │ DOCUMENT │  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                                             │
│  Min: 3 agents per phase  |  Max: 13 total  |  Parallel within phases      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, Send
from langgraph.types import interrupt

# Phase-aware state with agent tracking
class OrchestratedState(TypedDict):
    current_phase: Literal["explore", "plan", "code", "test", "fix", "document"]
    active_agents: list[str]
    phase_context: dict  # Preserved across transitions
    completed_work: list[str]
    pending_items: list[str]
    key_decisions: list[dict]

# Phase subgraph factory
def create_phase_subgraph(phase: str, agent_configs: list[dict]) -> StateGraph:
    """Create a phase-specific subgraph with parallel agent execution."""
    graph = StateGraph(PhaseState)

    # Fan-out to specialized agents
    def route_to_agents(state):
        return [Send(agent["name"], state) for agent in agent_configs]

    graph.add_conditional_edges(START, route_to_agents)

    # Each agent node
    for config in agent_configs:
        graph.add_node(config["name"], create_agent_node(config))

    # Fan-in aggregation
    graph.add_node("aggregate", aggregate_phase_results)
    for config in agent_configs:
        graph.add_edge(config["name"], "aggregate")

    return graph.compile()

# Main orchestrator with phase transitions
def create_orchestrated_workflow():
    main_graph = StateGraph(OrchestratedState)

    phases = ["explore", "plan", "code", "test", "fix", "document"]
    for phase in phases:
        main_graph.add_node(phase, phase_subgraphs[phase])

    # Phase transition with context preservation
    for i, phase in enumerate(phases[:-1]):
        main_graph.add_edge(phase, phases[i+1])

    # Conditional loop: fix → test (if issues) or → document
    main_graph.add_conditional_edges("fix", should_retest_or_document)

    return main_graph.compile(checkpointer=checkpointer)
```

### Impact
- **3-5x parallelization** within each phase
- **Specialized expertise** per agent role
- **Clear phase gates** for quality control
- **Iterative fix loops** until tests pass

---

## 2. Distributed Lock Manager for Resource Coordination

### Current State
No resource locking exists—parallel validators could theoretically conflict on shared files.

### Proposed Enhancement
Implement the protocol's **five lock types** with ownership tracking:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOCK MANAGER ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Lock Types:                                                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────┐ │
│  │   FILE   │ │DIRECTORY │ │   TASK   │ │   API    │ │  DB   │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └───────┘ │
│                                                                 │
│  Features:                                                      │
│  • 5-minute timeout default                                     │
│  • Ownership tracking (who holds what)                          │
│  • Blocking query (who's blocking me?)                          │
│  • Context manager for guaranteed release                       │
│  • Deadlock detection                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import asyncio
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

@dataclass
class Lock:
    resource_type: Literal["file", "directory", "task", "api", "database"]
    resource_id: str
    owner_agent: str
    acquired_at: datetime
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=5))

    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.acquired_at + self.timeout

class LockManager:
    """Distributed lock manager for multi-agent resource coordination."""

    def __init__(self):
        self._locks: dict[str, Lock] = {}
        self._lock = asyncio.Lock()
        self._waiting: dict[str, list[str]] = {}  # resource -> waiting agents

    @asynccontextmanager
    async def acquire(
        self,
        resource_type: str,
        resource_id: str,
        agent_id: str,
        timeout: timedelta = timedelta(minutes=5)
    ):
        """Context manager for safe lock acquisition and release."""
        key = f"{resource_type}:{resource_id}"

        async with self._lock:
            # Check for expired locks
            if key in self._locks and self._locks[key].is_expired:
                del self._locks[key]

            # Wait for lock if held
            while key in self._locks:
                holder = self._locks[key].owner_agent
                self._waiting.setdefault(key, []).append(agent_id)
                await asyncio.sleep(0.1)  # Yield control

                if self._locks.get(key) and self._locks[key].is_expired:
                    del self._locks[key]
                    break

            # Acquire lock
            self._locks[key] = Lock(
                resource_type=resource_type,
                resource_id=resource_id,
                owner_agent=agent_id,
                acquired_at=datetime.now(),
                timeout=timeout
            )

        try:
            yield
        finally:
            async with self._lock:
                if key in self._locks and self._locks[key].owner_agent == agent_id:
                    del self._locks[key]

    def who_holds(self, resource_type: str, resource_id: str) -> str | None:
        """Query which agent holds a lock."""
        key = f"{resource_type}:{resource_id}"
        lock = self._locks.get(key)
        return lock.owner_agent if lock and not lock.is_expired else None

# Integration with LangGraph nodes
lock_manager = LockManager()

async def generate_code_node_with_locking(state: TemplateGenerationState):
    output_dir = state["output_dir"]
    agent_id = state.get("current_agent", "generator")

    async with lock_manager.acquire("directory", output_dir, agent_id):
        # Safe to write to output_dir
        result = await generate_template(state)

    return {"generation_result": result}
```

### Impact
- **Prevents race conditions** in parallel agent execution
- **Enables safe file operations** during concurrent validation
- **Supports complex workflows** with shared resources
- **Provides debugging visibility** into resource contention

---

## 3. Inter-Agent Communication Channels

### Current State
Agents communicate only through shared state—no direct messaging capability.

### Proposed Enhancement
Implement the protocol's **three-tier communication system**:

```
┌─────────────────────────────────────────────────────────────────┐
│                   COMMUNICATION CHANNELS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐                                            │
│  │   BROADCAST     │  All agents receive (announcements)        │
│  │   ═══════════   │  "Template selection complete: copier"     │
│  └─────────────────┘                                            │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │    DIRECT       │  Point-to-point (requests/responses)       │
│  │   ──────────→   │  Researcher → Planner: "Found 3 deps"      │
│  └─────────────────┘                                            │
│                                                                 │
│  ┌─────────────────┐                                            │
│  │  TASK-SCOPED    │  Collaboration on work items               │
│  │   ◇──────◇      │  All code-phase agents on "api_routes"     │
│  └─────────────────┘                                            │
│                                                                 │
│  Message Types: request | response | notification | status |    │
│                 error | handoff                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any
import asyncio

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    STATUS = "status"
    ERROR = "error"
    HANDOFF = "handoff"

class ChannelType(Enum):
    BROADCAST = "broadcast"
    DIRECT = "direct"
    TASK_SCOPED = "task_scoped"

@dataclass
class AgentMessage:
    type: MessageType
    channel: ChannelType
    sender: str
    content: Any
    recipients: list[str] | None = None  # None = broadcast
    task_id: str | None = None  # For task-scoped channels
    timestamp: datetime = field(default_factory=datetime.now)

class MessageBus:
    """Central message bus for inter-agent communication."""

    def __init__(self):
        self._queues: dict[str, asyncio.Queue] = {}
        self._task_channels: dict[str, set[str]] = {}  # task_id -> subscribed agents
        self._message_history: list[AgentMessage] = []

    def register_agent(self, agent_id: str):
        """Register an agent to receive messages."""
        self._queues[agent_id] = asyncio.Queue()

    async def broadcast(self, sender: str, content: Any, msg_type: MessageType = MessageType.NOTIFICATION):
        """Send message to all agents."""
        msg = AgentMessage(
            type=msg_type,
            channel=ChannelType.BROADCAST,
            sender=sender,
            content=content
        )
        self._message_history.append(msg)

        for agent_id, queue in self._queues.items():
            if agent_id != sender:
                await queue.put(msg)

    async def send_direct(self, sender: str, recipient: str, content: Any, msg_type: MessageType = MessageType.REQUEST):
        """Send message to specific agent."""
        msg = AgentMessage(
            type=msg_type,
            channel=ChannelType.DIRECT,
            sender=sender,
            recipients=[recipient],
            content=content
        )
        self._message_history.append(msg)

        if recipient in self._queues:
            await self._queues[recipient].put(msg)

    def subscribe_to_task(self, agent_id: str, task_id: str):
        """Subscribe agent to task-scoped channel."""
        self._task_channels.setdefault(task_id, set()).add(agent_id)

    async def send_to_task(self, sender: str, task_id: str, content: Any):
        """Send message to all agents working on a task."""
        msg = AgentMessage(
            type=MessageType.NOTIFICATION,
            channel=ChannelType.TASK_SCOPED,
            sender=sender,
            task_id=task_id,
            content=content
        )
        self._message_history.append(msg)

        for agent_id in self._task_channels.get(task_id, []):
            if agent_id != sender and agent_id in self._queues:
                await self._queues[agent_id].put(msg)

    async def receive(self, agent_id: str, timeout: float = 0.1) -> AgentMessage | None:
        """Check for incoming messages (non-blocking by default)."""
        try:
            return await asyncio.wait_for(
                self._queues[agent_id].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None

# Integration with LangGraph state
class CommunicatingState(TypedDict):
    message_bus: MessageBus  # Shared across all nodes
    pending_messages: Annotated[list[AgentMessage], operator.add]
    agent_id: str

async def agent_node_with_communication(state: CommunicatingState):
    bus = state["message_bus"]
    agent_id = state["agent_id"]

    # Check for incoming messages before acting
    while msg := await bus.receive(agent_id):
        if msg.type == MessageType.REQUEST:
            # Handle request
            pass

    # Announce intention before acting
    await bus.broadcast(agent_id, f"Starting template analysis", MessageType.STATUS)

    # Do work...
    result = await analyze_template(state)

    # Handoff to next agent
    await bus.send_direct(agent_id, "planner", result, MessageType.HANDOFF)

    return {"analysis_result": result}
```

### Impact
- **Enables complex coordination** patterns beyond state sharing
- **Supports intention announcement** to prevent conflicts
- **Allows task-specific collaboration** groups
- **Provides message history** for debugging and audit

---

## 4. Advanced Checkpointing with Recovery & Rollback

### Current State
Basic `MemorySaver` checkpointing exists but lacks:
- Periodic checkpoints
- Pre-risky operation snapshots
- File state capture
- Recovery procedures

### Proposed Enhancement
Implement the protocol's **comprehensive checkpoint system**:

```
┌─────────────────────────────────────────────────────────────────┐
│                  CHECKPOINT ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Checkpoint Triggers:                                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  TASK    │ │  PHASE   │ │PRE-RISKY │ │  TIMED   │           │
│  │  START   │ │TRANSITION│ │OPERATION │ │ (5 min)  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│       ↓            ↓            ↓            ↓                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    CHECKPOINT DATA                          ││
│  │  • Task metadata & progress          • File snapshots       ││
│  │  • Execution context                 • Agent states         ││
│  │  • Key decisions log                 • Pending items        ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓                                  │
│  Recovery: restore_checkpoint(checkpoint_id) → resume execution │
│  Rollback: rollback_to(checkpoint_id) → undo changes            │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
import json
import shutil
from pathlib import Path
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver

@dataclass
class EnhancedCheckpoint:
    checkpoint_id: str
    timestamp: datetime
    trigger: Literal["task_start", "phase_transition", "pre_risky", "periodic", "error", "task_complete"]

    # State data
    task_metadata: dict
    progress: dict
    execution_context: dict

    # File snapshots
    file_snapshots: dict[str, str]  # path -> content hash
    snapshot_dir: Path | None = None

    # Recovery info
    key_decisions: list[dict]
    pending_items: list[str]
    agent_states: dict[str, dict]

class EnhancedCheckpointer:
    """Production-grade checkpointing with file snapshots and recovery."""

    def __init__(self, storage_dir: Path = Path(".checkpoints")):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(exist_ok=True)
        self._base_checkpointer = MemorySaver()
        self._checkpoints: dict[str, EnhancedCheckpoint] = {}
        self._last_periodic: datetime | None = None
        self.periodic_interval = timedelta(minutes=5)

    async def checkpoint(
        self,
        state: dict,
        trigger: str,
        output_dir: Path | None = None
    ) -> str:
        """Create a comprehensive checkpoint."""
        checkpoint_id = f"{trigger}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Snapshot files if output directory exists
        file_snapshots = {}
        snapshot_dir = None
        if output_dir and output_dir.exists():
            snapshot_dir = self.storage_dir / checkpoint_id / "files"
            snapshot_dir.mkdir(parents=True)
            shutil.copytree(output_dir, snapshot_dir / output_dir.name)

            for file in output_dir.rglob("*"):
                if file.is_file():
                    file_snapshots[str(file)] = self._hash_file(file)

        checkpoint = EnhancedCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            trigger=trigger,
            task_metadata=state.get("task_metadata", {}),
            progress={
                "current_phase": state.get("current_phase"),
                "completed_nodes": state.get("completed_nodes", []),
                "retry_count": state.get("retry_count", 0)
            },
            execution_context={
                "template_type": state.get("template_type"),
                "template_name": state.get("template_name"),
                "context_variables": state.get("context_variables", {})
            },
            file_snapshots=file_snapshots,
            snapshot_dir=snapshot_dir,
            key_decisions=state.get("key_decisions", []),
            pending_items=state.get("pending_items", []),
            agent_states=state.get("agent_states", {})
        )

        self._checkpoints[checkpoint_id] = checkpoint
        self._save_checkpoint(checkpoint)

        return checkpoint_id

    async def maybe_periodic_checkpoint(self, state: dict) -> str | None:
        """Create periodic checkpoint if interval elapsed."""
        now = datetime.now()
        if self._last_periodic is None or (now - self._last_periodic) > self.periodic_interval:
            self._last_periodic = now
            return await self.checkpoint(state, "periodic")
        return None

    async def restore(self, checkpoint_id: str) -> dict:
        """Restore state from checkpoint."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            checkpoint = self._load_checkpoint(checkpoint_id)

        return {
            "current_phase": checkpoint.progress["current_phase"],
            "completed_nodes": checkpoint.progress["completed_nodes"],
            "retry_count": checkpoint.progress["retry_count"],
            "template_type": checkpoint.execution_context["template_type"],
            "template_name": checkpoint.execution_context["template_name"],
            "context_variables": checkpoint.execution_context["context_variables"],
            "key_decisions": checkpoint.key_decisions,
            "pending_items": checkpoint.pending_items,
            "_restored_from": checkpoint_id
        }

    async def rollback(self, checkpoint_id: str, output_dir: Path) -> dict:
        """Rollback to checkpoint, restoring files."""
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint.snapshot_dir:
            # Remove current output
            if output_dir.exists():
                shutil.rmtree(output_dir)
            # Restore snapshot
            shutil.copytree(
                checkpoint.snapshot_dir / output_dir.name,
                output_dir
            )

        return await self.restore(checkpoint_id)

# Decorator for risky operations
def checkpoint_before_risky(checkpointer: EnhancedCheckpointer):
    def decorator(func):
        async def wrapper(state, *args, **kwargs):
            output_dir = state.get("output_dir")
            await checkpointer.checkpoint(state, "pre_risky", Path(output_dir) if output_dir else None)
            try:
                return await func(state, *args, **kwargs)
            except Exception as e:
                await checkpointer.checkpoint({**state, "error": str(e)}, "error")
                raise
        return wrapper
    return decorator

# Usage in nodes
checkpointer = EnhancedCheckpointer()

@checkpoint_before_risky(checkpointer)
async def generate_code_node(state):
    """Generate code with automatic pre-checkpoint."""
    # Risky file operations here...
    pass
```

### Impact
- **Zero work loss** on failures via checkpoint restoration
- **Rollback capability** for undoing bad generations
- **Audit trail** of all major state transitions
- **File-level recovery** not just state recovery

---

## 5. Context Preservation Across Phase Transitions

### Current State
State is passed between nodes but there's no explicit context preservation protocol for phase transitions.

### Proposed Enhancement
Implement the protocol's **mandatory context preservation** requirements:

```
┌─────────────────────────────────────────────────────────────────┐
│              CONTEXT PRESERVATION PROTOCOL                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase Transition: EXPLORE → PLAN                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  HANDOFF CONTEXT (Mandatory Fields)                         ││
│  │                                                             ││
│  │  ✓ phase_id: "explore"                                      ││
│  │  ✓ active_agents: ["researcher", "analyzer"]                ││
│  │  ✓ completed_work: ["dependency analysis", "codebase scan"] ││
│  │  ✓ pending_items: ["security review"]                       ││
│  │  ✓ key_decisions: [{decision, rationale, alternatives}]     ││
│  │  ✓ file_locations: {"config": "src/config.py", ...}         ││
│  │  ✓ dependencies: ["langchain", "pydantic"]                  ││
│  │  ✓ issues_found: ["deprecated API usage"]                   ││
│  │  ✓ next_steps: ["create implementation plan", ...]          ││
│  └─────────────────────────────────────────────────────────────┘│
│                              ↓                                  │
│                         VALIDATION                              │
│                   (all fields required)                         │
│                              ↓                                  │
│                    PLAN PHASE BEGINS                            │
│           (full context available to planners)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from pydantic import BaseModel, validator
from typing import Optional

class KeyDecision(BaseModel):
    """Record of an important decision made during execution."""
    decision: str
    rationale: str
    alternatives_considered: list[str]
    made_by: str  # agent_id
    phase: str
    timestamp: datetime = Field(default_factory=datetime.now)

class PhaseHandoffContext(BaseModel):
    """Mandatory context for phase transitions - prevents knowledge loss."""

    # Phase identification
    from_phase: str
    to_phase: str
    active_agents: list[str]

    # Work tracking
    completed_work: list[str]
    pending_items: list[str]

    # Knowledge preservation
    key_decisions: list[KeyDecision]
    file_locations: dict[str, str]  # logical name -> path
    dependencies_discovered: list[str]

    # Issues and next steps
    issues_requiring_attention: list[str]
    explicit_next_steps: list[str]

    # Optional context
    code_snippets: dict[str, str] = {}  # name -> code
    external_references: list[str] = []

    @validator('completed_work', 'explicit_next_steps')
    def must_not_be_empty(cls, v):
        if not v:
            raise ValueError("Context preservation requires explicit tracking")
        return v

class ContextPreservingState(TypedDict):
    """State schema with mandatory context preservation."""
    # Current phase info
    current_phase: str
    phase_history: list[PhaseHandoffContext]

    # Live context (updated throughout phase)
    active_agents: list[str]
    completed_work: Annotated[list[str], operator.add]
    pending_items: list[str]
    key_decisions: Annotated[list[KeyDecision], operator.add]

    # Discovery tracking
    file_locations: dict[str, str]
    dependencies: list[str]
    issues: list[str]

    # Explicit next steps (required)
    next_steps: list[str]

def create_phase_transition_edge(from_phase: str, to_phase: str):
    """Create an edge that enforces context preservation."""

    def transition_node(state: ContextPreservingState) -> ContextPreservingState:
        # Validate required context exists
        handoff = PhaseHandoffContext(
            from_phase=from_phase,
            to_phase=to_phase,
            active_agents=state["active_agents"],
            completed_work=state["completed_work"],
            pending_items=state["pending_items"],
            key_decisions=state["key_decisions"],
            file_locations=state["file_locations"],
            dependencies_discovered=state["dependencies"],
            issues_requiring_attention=state["issues"],
            explicit_next_steps=state["next_steps"]
        )

        # Log transition
        logger.info(f"Phase transition: {from_phase} → {to_phase}")
        logger.info(f"Completed: {handoff.completed_work}")
        logger.info(f"Pending: {handoff.pending_items}")
        logger.info(f"Key decisions: {len(handoff.key_decisions)}")

        return {
            **state,
            "current_phase": to_phase,
            "phase_history": state.get("phase_history", []) + [handoff],
            # Reset for new phase
            "active_agents": [],
            "pending_items": handoff.explicit_next_steps,  # Next steps become pending
        }

    return transition_node

# Helper for agents to record decisions
def record_decision(
    state: ContextPreservingState,
    decision: str,
    rationale: str,
    alternatives: list[str],
    agent_id: str
) -> ContextPreservingState:
    """Record a key decision with full context."""
    return {
        **state,
        "key_decisions": state["key_decisions"] + [KeyDecision(
            decision=decision,
            rationale=rationale,
            alternatives_considered=alternatives,
            made_by=agent_id,
            phase=state["current_phase"]
        )]
    }

# Example usage in explore phase
async def researcher_node(state: ContextPreservingState):
    """Researcher agent with context preservation."""

    # Do research...
    found_deps = ["langchain", "pydantic", "fastapi"]
    found_files = {"main_entry": "src/main.py", "config": "src/config.py"}

    # Record findings
    state = record_decision(
        state,
        decision="Use FastAPI for API layer",
        rationale="Already in dependencies, good async support",
        alternatives=["Flask", "Starlette", "Django"],
        agent_id="researcher"
    )

    return {
        **state,
        "completed_work": ["dependency analysis", "codebase structure mapping"],
        "dependencies": found_deps,
        "file_locations": {**state["file_locations"], **found_files},
        "next_steps": [
            "Create implementation plan based on discovered structure",
            "Design API routes for template generation",
            "Plan database schema for template metadata"
        ]
    }
```

### Impact
- **Zero knowledge loss** between phases
- **Full audit trail** of decisions and rationale
- **Clear accountability** (which agent decided what)
- **Explicit next steps** prevent work being forgotten
- **Validation** ensures no incomplete handoffs

---

## Summary: Implementation Complete

| Idea | Impact | Complexity | Status |
|------|--------|------------|--------|
| 1. Six-Phase Multi-Agent | Very High | High | ✅ Implemented |
| 2. Lock Manager | High | Medium | ✅ Implemented |
| 3. Communication Channels | High | Medium | ✅ Implemented |
| 4. Advanced Checkpointing | Very High | Medium | ✅ Implemented |
| 5. Context Preservation | High | Low | ✅ Implemented |

### Implementation Files

- **Main Module**: `langgang/orchestration.py` (~1000 lines)
- **Exports**: `langgang/__init__.py` (updated with all orchestration classes)
- **Version**: Bumped to 0.3.0

### API Conformance

The implementation follows LangGraph specifications from:
- [LangGraph Graph API](https://reference.langchain.com/python/langgraph/graphs/)
- [LangGraph Pregel](https://reference.langchain.com/python/langgraph/pregel/)
- [Deep Agents Middleware](https://reference.langchain.com/python/deepagents/middleware/)
- [Deep Agents Graph](https://reference.langchain.com/python/deepagents/graph/)

---

## References

- [Claude Orchestration Protocol v4.0.0](https://github.com/Lobbi-Docs/claude/blob/main/.claude/orchestration/PROTOCOL.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Patterns: Fan-Out/Fan-In](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [LangChain Reference](https://reference.langchain.com/python/)
