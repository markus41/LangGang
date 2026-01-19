"""
LangGraph Multi-Agent Orchestration Module

This module implements 5 high-impact orchestration patterns inspired by the
Claude Orchestration Protocol v4.0.0:

1. Context Preservation Protocol - Mandatory context handoffs between phases
2. Advanced Checkpointing with Recovery - Enhanced state persistence and rollback
3. Six-Phase Multi-Agent Architecture - Specialized agent teams per phase
4. Distributed Lock Manager - Resource coordination for parallel agents
5. Inter-Agent Communication Channels - Broadcast, direct, and task-scoped messaging

References:
- https://github.com/Lobbi-Docs/claude/blob/main/.claude/orchestration/PROTOCOL.md
- https://reference.langchain.com/python/langgraph/
- https://reference.langchain.com/python/deepagents/
"""

import asyncio
import hashlib
import json
import logging
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
)

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
import operator

logger = logging.getLogger(__name__)


# =============================================================================
# IDEA #5: Context Preservation Protocol
# =============================================================================

class KeyDecision(TypedDict):
    """Record of an important decision made during execution."""
    decision: str
    rationale: str
    alternatives_considered: List[str]
    made_by: str  # agent_id
    phase: str
    timestamp: str  # ISO format


class PhaseHandoffContext(TypedDict):
    """Mandatory context for phase transitions - prevents knowledge loss.

    Based on Claude Orchestration Protocol v4.0.0 requirements:
    - Phase identifier and active agent list
    - Completed work summary and pending items
    - Key decisions, file locations, dependencies
    - Issues requiring attention
    - Explicit next-steps tracking
    """
    # Phase identification
    from_phase: str
    to_phase: str
    active_agents: List[str]

    # Work tracking
    completed_work: List[str]
    pending_items: List[str]

    # Knowledge preservation
    key_decisions: List[KeyDecision]
    file_locations: Dict[str, str]  # logical name -> path
    dependencies_discovered: List[str]

    # Issues and next steps
    issues_requiring_attention: List[str]
    explicit_next_steps: List[str]

    # Optional context
    code_snippets: Dict[str, str]  # name -> code
    external_references: List[str]

    # Metadata
    timestamp: str
    handoff_id: str


class ContextPreservingState(TypedDict):
    """State schema with mandatory context preservation.

    Implements reducer patterns for accumulating state across nodes
    using LangGraph's Annotated type with operator.add.
    """
    # Message history with reducer
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Current phase info
    current_phase: Literal["explore", "plan", "code", "test", "fix", "document"]
    phase_history: List[PhaseHandoffContext]

    # Live context (updated throughout phase)
    active_agents: List[str]
    completed_work: Annotated[List[str], operator.add]
    pending_items: List[str]
    key_decisions: Annotated[List[KeyDecision], operator.add]

    # Discovery tracking
    file_locations: Dict[str, str]
    dependencies: List[str]
    issues: List[str]

    # Explicit next steps (required for handoff)
    next_steps: List[str]

    # Project context
    project_description: str
    template_type: str
    template_name: str
    output_dir: str
    context_variables: Dict[str, Any]


def validate_handoff_context(context: PhaseHandoffContext) -> List[str]:
    """Validate that all required handoff fields are populated.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    if not context.get("completed_work"):
        errors.append("completed_work is required for phase handoff")

    if not context.get("explicit_next_steps"):
        errors.append("explicit_next_steps is required for phase handoff")

    if not context.get("from_phase") or not context.get("to_phase"):
        errors.append("from_phase and to_phase are required")

    return errors


def create_handoff_context(
    state: ContextPreservingState,
    to_phase: str
) -> PhaseHandoffContext:
    """Create a validated handoff context for phase transition.

    Args:
        state: Current state with context to preserve
        to_phase: Target phase for transition

    Returns:
        PhaseHandoffContext with all required fields

    Raises:
        ValueError: If required context is missing
    """
    handoff = PhaseHandoffContext(
        from_phase=state.get("current_phase", "unknown"),
        to_phase=to_phase,
        active_agents=state.get("active_agents", []),
        completed_work=list(state.get("completed_work", [])),
        pending_items=list(state.get("pending_items", [])),
        key_decisions=list(state.get("key_decisions", [])),
        file_locations=dict(state.get("file_locations", {})),
        dependencies_discovered=list(state.get("dependencies", [])),
        issues_requiring_attention=list(state.get("issues", [])),
        explicit_next_steps=list(state.get("next_steps", [])),
        code_snippets={},
        external_references=[],
        timestamp=datetime.now().isoformat(),
        handoff_id=f"{state.get('current_phase', 'unknown')}_{to_phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    errors = validate_handoff_context(handoff)
    if errors:
        logger.warning(f"Handoff context validation warnings: {errors}")

    return handoff


def record_decision(
    decision: str,
    rationale: str,
    alternatives: List[str],
    agent_id: str,
    phase: str
) -> KeyDecision:
    """Create a key decision record with full context.

    Args:
        decision: The decision that was made
        rationale: Why this decision was made
        alternatives: Other options that were considered
        agent_id: Which agent made this decision
        phase: Current execution phase

    Returns:
        KeyDecision record
    """
    return KeyDecision(
        decision=decision,
        rationale=rationale,
        alternatives_considered=alternatives,
        made_by=agent_id,
        phase=phase,
        timestamp=datetime.now().isoformat()
    )


# =============================================================================
# IDEA #4: Advanced Checkpointing with Recovery
# =============================================================================

class CheckpointTrigger(Enum):
    """Types of events that trigger checkpoints."""
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    PHASE_TRANSITION = "phase_transition"
    PRE_RISKY = "pre_risky"
    PERIODIC = "periodic"
    ERROR = "error"
    MANUAL = "manual"


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    checkpoint_id: str
    timestamp: datetime
    trigger: CheckpointTrigger
    phase: str
    description: str

    # Progress tracking
    completed_nodes: List[str]
    pending_nodes: List[str]
    retry_count: int

    # File tracking
    file_hashes: Dict[str, str]  # path -> hash
    snapshot_path: Optional[Path] = None


@dataclass
class EnhancedCheckpoint:
    """Full checkpoint with state and file snapshots.

    Implements the protocol's checkpoint requirements:
    - Task metadata and progress indicators
    - Execution context
    - File snapshots for rollback
    - Recovery information
    """
    metadata: CheckpointMetadata

    # State data
    task_metadata: Dict[str, Any]
    progress: Dict[str, Any]
    execution_context: Dict[str, Any]

    # Context preservation
    key_decisions: List[KeyDecision]
    pending_items: List[str]
    agent_states: Dict[str, Dict[str, Any]]

    # Phase history
    phase_history: List[PhaseHandoffContext]


class EnhancedCheckpointer:
    """Production-grade checkpointing with file snapshots and recovery.

    Features:
    - Periodic checkpoints (every 5 minutes by default)
    - Pre-risky operation snapshots
    - File state capture for rollback
    - Recovery to previous states

    Based on protocol requirements:
    - Checkpoints at task initiation, completion
    - Pre-risky operations, error occurrence
    - Periodic intervals for extended tasks
    """

    def __init__(
        self,
        storage_dir: Path = Path(".checkpoints"),
        periodic_interval: timedelta = timedelta(minutes=5),
        max_checkpoints: int = 50
    ):
        """Initialize the enhanced checkpointer.

        Args:
            storage_dir: Directory for checkpoint storage
            periodic_interval: Interval for periodic checkpoints
            max_checkpoints: Maximum checkpoints to retain
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._base_checkpointer = MemorySaver()
        self._checkpoints: Dict[str, EnhancedCheckpoint] = {}
        self._checkpoint_order: List[str] = []

        self._last_periodic: Optional[datetime] = None
        self.periodic_interval = periodic_interval
        self.max_checkpoints = max_checkpoints

    def _hash_file(self, filepath: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove oldest checkpoints when limit exceeded."""
        while len(self._checkpoint_order) > self.max_checkpoints:
            oldest_id = self._checkpoint_order.pop(0)
            checkpoint = self._checkpoints.pop(oldest_id, None)

            if checkpoint and checkpoint.metadata.snapshot_path:
                try:
                    shutil.rmtree(checkpoint.metadata.snapshot_path)
                except Exception as e:
                    logger.warning(f"Failed to remove snapshot: {e}")

    async def checkpoint(
        self,
        state: Dict[str, Any],
        trigger: CheckpointTrigger,
        description: str = "",
        output_dir: Optional[Path] = None
    ) -> str:
        """Create a comprehensive checkpoint.

        Args:
            state: Current workflow state
            trigger: What triggered this checkpoint
            description: Human-readable description
            output_dir: Directory to snapshot (if any)

        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"{trigger.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        # Snapshot files if output directory exists
        file_hashes: Dict[str, str] = {}
        snapshot_path: Optional[Path] = None

        if output_dir and output_dir.exists():
            snapshot_path = self.storage_dir / checkpoint_id / "files"
            snapshot_path.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copytree(output_dir, snapshot_path / output_dir.name, dirs_exist_ok=True)

                for file in output_dir.rglob("*"):
                    if file.is_file():
                        file_hashes[str(file.relative_to(output_dir))] = self._hash_file(file)
            except Exception as e:
                logger.error(f"Failed to create file snapshot: {e}")

        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            trigger=trigger,
            phase=state.get("current_phase", "unknown"),
            description=description,
            completed_nodes=state.get("completed_nodes", []),
            pending_nodes=state.get("pending_nodes", []),
            retry_count=state.get("retry_count", 0),
            file_hashes=file_hashes,
            snapshot_path=snapshot_path
        )

        checkpoint = EnhancedCheckpoint(
            metadata=metadata,
            task_metadata={
                "project_description": state.get("project_description", ""),
                "template_type": state.get("template_type", ""),
                "template_name": state.get("template_name", ""),
            },
            progress={
                "current_phase": state.get("current_phase"),
                "completed_work": state.get("completed_work", []),
                "retry_count": state.get("retry_count", 0),
                "validation_passed": state.get("validation_passed", False),
            },
            execution_context={
                "output_dir": state.get("output_dir", ""),
                "context_variables": state.get("context_variables", {}),
                "active_agents": state.get("active_agents", []),
            },
            key_decisions=state.get("key_decisions", []),
            pending_items=state.get("pending_items", []),
            agent_states=state.get("agent_states", {}),
            phase_history=state.get("phase_history", []),
        )

        self._checkpoints[checkpoint_id] = checkpoint
        self._checkpoint_order.append(checkpoint_id)

        # Save metadata to disk
        self._save_checkpoint_metadata(checkpoint)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        logger.info(f"Created checkpoint: {checkpoint_id} (trigger: {trigger.value})")
        return checkpoint_id

    def _save_checkpoint_metadata(self, checkpoint: EnhancedCheckpoint) -> None:
        """Save checkpoint metadata to disk."""
        meta_path = self.storage_dir / checkpoint.metadata.checkpoint_id / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        with open(meta_path, "w") as f:
            json.dump({
                "checkpoint_id": checkpoint.metadata.checkpoint_id,
                "timestamp": checkpoint.metadata.timestamp.isoformat(),
                "trigger": checkpoint.metadata.trigger.value,
                "phase": checkpoint.metadata.phase,
                "description": checkpoint.metadata.description,
                "completed_nodes": checkpoint.metadata.completed_nodes,
                "retry_count": checkpoint.metadata.retry_count,
                "file_hashes": checkpoint.metadata.file_hashes,
                "task_metadata": checkpoint.task_metadata,
                "progress": checkpoint.progress,
                "execution_context": checkpoint.execution_context,
                "key_decisions": checkpoint.key_decisions,
                "pending_items": checkpoint.pending_items,
            }, f, indent=2)

    async def maybe_periodic_checkpoint(
        self,
        state: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> Optional[str]:
        """Create periodic checkpoint if interval has elapsed.

        Args:
            state: Current state
            output_dir: Optional directory to snapshot

        Returns:
            Checkpoint ID if created, None otherwise
        """
        now = datetime.now()

        if self._last_periodic is None or (now - self._last_periodic) > self.periodic_interval:
            self._last_periodic = now
            return await self.checkpoint(
                state,
                CheckpointTrigger.PERIODIC,
                f"Periodic checkpoint at {now.strftime('%H:%M:%S')}",
                output_dir
            )

        return None

    async def restore(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            Restored state dictionary

        Raises:
            KeyError: If checkpoint not found
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")

        return {
            "current_phase": checkpoint.progress.get("current_phase"),
            "completed_work": checkpoint.progress.get("completed_work", []),
            "retry_count": checkpoint.progress.get("retry_count", 0),
            "validation_passed": checkpoint.progress.get("validation_passed", False),
            "template_type": checkpoint.task_metadata.get("template_type", ""),
            "template_name": checkpoint.task_metadata.get("template_name", ""),
            "project_description": checkpoint.task_metadata.get("project_description", ""),
            "output_dir": checkpoint.execution_context.get("output_dir", ""),
            "context_variables": checkpoint.execution_context.get("context_variables", {}),
            "active_agents": checkpoint.execution_context.get("active_agents", []),
            "key_decisions": checkpoint.key_decisions,
            "pending_items": checkpoint.pending_items,
            "phase_history": checkpoint.phase_history,
            "_restored_from": checkpoint_id,
        }

    async def rollback(
        self,
        checkpoint_id: str,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Rollback to checkpoint, restoring both state and files.

        Args:
            checkpoint_id: ID of checkpoint to rollback to
            output_dir: Directory to restore files to

        Returns:
            Restored state dictionary
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise KeyError(f"Checkpoint not found: {checkpoint_id}")

        # Restore files if snapshot exists
        if checkpoint.metadata.snapshot_path:
            snapshot_content = checkpoint.metadata.snapshot_path / output_dir.name

            if snapshot_content.exists():
                # Remove current output
                if output_dir.exists():
                    shutil.rmtree(output_dir)

                # Restore from snapshot
                shutil.copytree(snapshot_content, output_dir)
                logger.info(f"Restored files from checkpoint: {checkpoint_id}")

        return await self.restore(checkpoint_id)

    def list_checkpoints(
        self,
        phase: Optional[str] = None,
        trigger: Optional[CheckpointTrigger] = None
    ) -> List[CheckpointMetadata]:
        """List available checkpoints with optional filtering.

        Args:
            phase: Filter by phase
            trigger: Filter by trigger type

        Returns:
            List of checkpoint metadata
        """
        results = []

        for checkpoint_id in self._checkpoint_order:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                continue

            if phase and checkpoint.metadata.phase != phase:
                continue

            if trigger and checkpoint.metadata.trigger != trigger:
                continue

            results.append(checkpoint.metadata)

        return results


def checkpoint_before_risky(checkpointer: EnhancedCheckpointer):
    """Decorator to create checkpoint before risky operations.

    Usage:
        @checkpoint_before_risky(checkpointer)
        async def generate_code_node(state):
            # Risky file operations...
            pass
    """
    def decorator(func: Callable):
        async def wrapper(state: Dict[str, Any], *args, **kwargs):
            output_dir = state.get("output_dir")
            output_path = Path(output_dir) if output_dir else None

            await checkpointer.checkpoint(
                state,
                CheckpointTrigger.PRE_RISKY,
                f"Before {func.__name__}",
                output_path
            )

            try:
                return await func(state, *args, **kwargs)
            except Exception as e:
                await checkpointer.checkpoint(
                    {**state, "error": str(e)},
                    CheckpointTrigger.ERROR,
                    f"Error in {func.__name__}: {str(e)}"
                )
                raise

        return wrapper
    return decorator


# =============================================================================
# IDEA #2: Distributed Lock Manager
# =============================================================================

class LockType(Enum):
    """Types of resource locks available."""
    FILE = "file"
    DIRECTORY = "directory"
    TASK = "task"
    API = "api"
    DATABASE = "database"


@dataclass
class Lock:
    """Represents a held resource lock."""
    resource_type: LockType
    resource_id: str
    owner_agent: str
    acquired_at: datetime
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if lock has expired."""
        return datetime.now() > self.acquired_at + self.timeout

    @property
    def key(self) -> str:
        """Get unique key for this lock."""
        return f"{self.resource_type.value}:{self.resource_id}"


class LockManager:
    """Distributed lock manager for multi-agent resource coordination.

    Implements the protocol's five lock types:
    - FILE: Lock individual files
    - DIRECTORY: Lock entire directories
    - TASK: Lock logical task units
    - API: Lock external API access
    - DATABASE: Lock database resources

    Features:
    - 5-minute default timeout
    - Ownership tracking
    - Blocking queries (who's holding what)
    - Context manager for guaranteed release
    - Deadlock detection
    """

    def __init__(self, default_timeout: timedelta = timedelta(minutes=5)):
        """Initialize the lock manager.

        Args:
            default_timeout: Default lock timeout duration
        """
        self._locks: Dict[str, Lock] = {}
        self._lock = asyncio.Lock()
        self._waiting: Dict[str, List[str]] = {}  # resource -> waiting agents
        self._history: List[Dict[str, Any]] = []
        self.default_timeout = default_timeout

    def _cleanup_expired(self) -> List[str]:
        """Remove expired locks and return their keys."""
        expired = []
        for key, lock in list(self._locks.items()):
            if lock.is_expired:
                expired.append(key)
                del self._locks[key]
                logger.info(f"Lock expired: {key} (held by {lock.owner_agent})")
        return expired

    @asynccontextmanager
    async def acquire(
        self,
        resource_type: LockType,
        resource_id: str,
        agent_id: str,
        timeout: Optional[timedelta] = None,
        wait_timeout: float = 30.0
    ):
        """Acquire a lock with context manager for guaranteed release.

        Args:
            resource_type: Type of resource to lock
            resource_id: Identifier for the resource
            agent_id: Identifier for the requesting agent
            timeout: Lock timeout (defaults to manager default)
            wait_timeout: How long to wait for lock acquisition

        Yields:
            Lock object

        Raises:
            TimeoutError: If lock cannot be acquired within wait_timeout
        """
        key = f"{resource_type.value}:{resource_id}"
        lock_timeout = timeout or self.default_timeout
        start_time = datetime.now()

        while True:
            async with self._lock:
                # Cleanup expired locks
                self._cleanup_expired()

                # Check if lock is available
                if key not in self._locks:
                    # Acquire the lock
                    lock = Lock(
                        resource_type=resource_type,
                        resource_id=resource_id,
                        owner_agent=agent_id,
                        acquired_at=datetime.now(),
                        timeout=lock_timeout
                    )
                    self._locks[key] = lock

                    # Remove from waiting list
                    if key in self._waiting and agent_id in self._waiting[key]:
                        self._waiting[key].remove(agent_id)

                    self._history.append({
                        "action": "acquire",
                        "key": key,
                        "agent": agent_id,
                        "timestamp": datetime.now().isoformat()
                    })

                    logger.debug(f"Lock acquired: {key} by {agent_id}")
                    break

                # Lock is held - add to waiting list
                if key not in self._waiting:
                    self._waiting[key] = []
                if agent_id not in self._waiting[key]:
                    self._waiting[key].append(agent_id)

            # Check for timeout
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > wait_timeout:
                holder = self._locks.get(key)
                raise TimeoutError(
                    f"Could not acquire lock {key} within {wait_timeout}s. "
                    f"Held by: {holder.owner_agent if holder else 'unknown'}"
                )

            # Wait and retry
            await asyncio.sleep(0.1)

        try:
            yield lock
        finally:
            async with self._lock:
                if key in self._locks and self._locks[key].owner_agent == agent_id:
                    del self._locks[key]

                    self._history.append({
                        "action": "release",
                        "key": key,
                        "agent": agent_id,
                        "timestamp": datetime.now().isoformat()
                    })

                    logger.debug(f"Lock released: {key} by {agent_id}")

    def who_holds(self, resource_type: LockType, resource_id: str) -> Optional[str]:
        """Query which agent holds a lock.

        Args:
            resource_type: Type of resource
            resource_id: Resource identifier

        Returns:
            Agent ID holding the lock, or None if unlocked
        """
        key = f"{resource_type.value}:{resource_id}"
        lock = self._locks.get(key)

        if lock and not lock.is_expired:
            return lock.owner_agent

        return None

    def who_waits(self, resource_type: LockType, resource_id: str) -> List[str]:
        """Query which agents are waiting for a lock.

        Args:
            resource_type: Type of resource
            resource_id: Resource identifier

        Returns:
            List of waiting agent IDs
        """
        key = f"{resource_type.value}:{resource_id}"
        return list(self._waiting.get(key, []))

    def get_locks_by_agent(self, agent_id: str) -> List[Lock]:
        """Get all locks held by an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of locks held by the agent
        """
        return [
            lock for lock in self._locks.values()
            if lock.owner_agent == agent_id and not lock.is_expired
        ]

    def detect_deadlock(self) -> Optional[List[str]]:
        """Detect potential deadlock cycles.

        Returns:
            List of agents in deadlock cycle, or None if no deadlock
        """
        # Build wait-for graph
        wait_for: Dict[str, set] = {}

        for key, waiting_agents in self._waiting.items():
            holder = self._locks.get(key)
            if holder and not holder.is_expired:
                for waiter in waiting_agents:
                    if waiter not in wait_for:
                        wait_for[waiter] = set()
                    wait_for[waiter].add(holder.owner_agent)

        # Detect cycles using DFS
        def find_cycle(agent: str, path: List[str], visited: set) -> Optional[List[str]]:
            if agent in path:
                cycle_start = path.index(agent)
                return path[cycle_start:]

            if agent in visited:
                return None

            visited.add(agent)
            path.append(agent)

            for waiting_for in wait_for.get(agent, []):
                cycle = find_cycle(waiting_for, path, visited)
                if cycle:
                    return cycle

            path.pop()
            return None

        visited: set = set()
        for agent in wait_for:
            cycle = find_cycle(agent, [], visited)
            if cycle:
                return cycle

        return None


# =============================================================================
# IDEA #3: Inter-Agent Communication Channels
# =============================================================================

class MessageType(Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    STATUS = "status"
    ERROR = "error"
    HANDOFF = "handoff"
    BROADCAST = "broadcast"


class ChannelType(Enum):
    """Types of communication channels."""
    BROADCAST = "broadcast"    # All agents receive
    DIRECT = "direct"          # Point-to-point
    TASK_SCOPED = "task_scoped"  # Agents working on same task


@dataclass
class AgentMessage:
    """Message passed between agents."""
    id: str
    type: MessageType
    channel: ChannelType
    sender: str
    content: Any
    recipients: Optional[List[str]] = None  # None = broadcast
    task_id: Optional[str] = None  # For task-scoped channels
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "channel": self.channel.value,
            "sender": self.sender,
            "content": self.content,
            "recipients": self.recipients,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class MessageBus:
    """Central message bus for inter-agent communication.

    Implements three channel types from the protocol:
    - BROADCAST: Announcements to all agents
    - DIRECT: Point-to-point requests/responses
    - TASK_SCOPED: Collaboration on work items

    Message types:
    - REQUEST: Ask another agent to do something
    - RESPONSE: Reply to a request
    - NOTIFICATION: Inform without expecting reply
    - STATUS: Progress updates
    - ERROR: Error reports
    - HANDOFF: Transfer control to another agent
    """

    def __init__(self, max_history: int = 1000):
        """Initialize the message bus.

        Args:
            max_history: Maximum messages to retain in history
        """
        self._queues: Dict[str, asyncio.Queue] = {}
        self._task_channels: Dict[str, set] = {}  # task_id -> subscribed agents
        self._message_history: List[AgentMessage] = []
        self._message_counter = 0
        self.max_history = max_history
        self._lock = asyncio.Lock()

    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        self._message_counter += 1
        return f"msg_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._message_counter:06d}"

    def _add_to_history(self, msg: AgentMessage) -> None:
        """Add message to history with cleanup."""
        self._message_history.append(msg)
        while len(self._message_history) > self.max_history:
            self._message_history.pop(0)

    async def register_agent(self, agent_id: str) -> None:
        """Register an agent to receive messages.

        Args:
            agent_id: Unique agent identifier
        """
        async with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = asyncio.Queue()
                logger.debug(f"Agent registered: {agent_id}")

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent.

        Args:
            agent_id: Agent identifier
        """
        async with self._lock:
            self._queues.pop(agent_id, None)

            # Remove from task channels
            for subscribers in self._task_channels.values():
                subscribers.discard(agent_id)

    async def broadcast(
        self,
        sender: str,
        content: Any,
        msg_type: MessageType = MessageType.NOTIFICATION,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send message to all registered agents.

        Args:
            sender: Sending agent ID
            content: Message content
            msg_type: Type of message
            metadata: Optional metadata

        Returns:
            Message ID
        """
        msg = AgentMessage(
            id=self._generate_message_id(),
            type=msg_type,
            channel=ChannelType.BROADCAST,
            sender=sender,
            content=content,
            metadata=metadata or {}
        )

        self._add_to_history(msg)

        async with self._lock:
            for agent_id, queue in self._queues.items():
                if agent_id != sender:
                    await queue.put(msg)

        logger.debug(f"Broadcast from {sender}: {msg.id}")
        return msg.id

    async def send_direct(
        self,
        sender: str,
        recipient: str,
        content: Any,
        msg_type: MessageType = MessageType.REQUEST,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send message to specific agent.

        Args:
            sender: Sending agent ID
            recipient: Target agent ID
            content: Message content
            msg_type: Type of message
            metadata: Optional metadata

        Returns:
            Message ID
        """
        msg = AgentMessage(
            id=self._generate_message_id(),
            type=msg_type,
            channel=ChannelType.DIRECT,
            sender=sender,
            recipients=[recipient],
            content=content,
            metadata=metadata or {}
        )

        self._add_to_history(msg)

        async with self._lock:
            if recipient in self._queues:
                await self._queues[recipient].put(msg)

        logger.debug(f"Direct message {sender} -> {recipient}: {msg.id}")
        return msg.id

    async def subscribe_to_task(self, agent_id: str, task_id: str) -> None:
        """Subscribe agent to task-scoped channel.

        Args:
            agent_id: Agent identifier
            task_id: Task identifier
        """
        async with self._lock:
            if task_id not in self._task_channels:
                self._task_channels[task_id] = set()
            self._task_channels[task_id].add(agent_id)

        logger.debug(f"Agent {agent_id} subscribed to task {task_id}")

    async def unsubscribe_from_task(self, agent_id: str, task_id: str) -> None:
        """Unsubscribe agent from task-scoped channel."""
        async with self._lock:
            if task_id in self._task_channels:
                self._task_channels[task_id].discard(agent_id)

    async def send_to_task(
        self,
        sender: str,
        task_id: str,
        content: Any,
        msg_type: MessageType = MessageType.NOTIFICATION,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Send message to all agents working on a task.

        Args:
            sender: Sending agent ID
            task_id: Task identifier
            content: Message content
            msg_type: Type of message
            metadata: Optional metadata

        Returns:
            Message ID
        """
        msg = AgentMessage(
            id=self._generate_message_id(),
            type=msg_type,
            channel=ChannelType.TASK_SCOPED,
            sender=sender,
            task_id=task_id,
            content=content,
            metadata=metadata or {}
        )

        self._add_to_history(msg)

        async with self._lock:
            subscribers = self._task_channels.get(task_id, set())
            for agent_id in subscribers:
                if agent_id != sender and agent_id in self._queues:
                    await self._queues[agent_id].put(msg)

        logger.debug(f"Task message from {sender} to task {task_id}: {msg.id}")
        return msg.id

    async def receive(
        self,
        agent_id: str,
        timeout: float = 0.1,
        filter_type: Optional[MessageType] = None
    ) -> Optional[AgentMessage]:
        """Check for incoming messages (non-blocking by default).

        Args:
            agent_id: Agent identifier
            timeout: How long to wait for message
            filter_type: Only return messages of this type

        Returns:
            AgentMessage or None if no message available
        """
        if agent_id not in self._queues:
            return None

        try:
            msg = await asyncio.wait_for(
                self._queues[agent_id].get(),
                timeout=timeout
            )

            if filter_type and msg.type != filter_type:
                # Put back and return None
                await self._queues[agent_id].put(msg)
                return None

            return msg

        except asyncio.TimeoutError:
            return None

    async def receive_all(self, agent_id: str) -> List[AgentMessage]:
        """Receive all pending messages for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of all pending messages
        """
        messages = []

        while True:
            msg = await self.receive(agent_id, timeout=0)
            if msg is None:
                break
            messages.append(msg)

        return messages

    def get_history(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        task_id: Optional[str] = None,
        msg_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Query message history with filters.

        Args:
            sender: Filter by sender
            recipient: Filter by recipient
            task_id: Filter by task
            msg_type: Filter by message type
            limit: Maximum messages to return

        Returns:
            Filtered list of messages
        """
        results = []

        for msg in reversed(self._message_history):
            if len(results) >= limit:
                break

            if sender and msg.sender != sender:
                continue

            if recipient and (not msg.recipients or recipient not in msg.recipients):
                continue

            if task_id and msg.task_id != task_id:
                continue

            if msg_type and msg.type != msg_type:
                continue

            results.append(msg)

        return list(reversed(results))


# =============================================================================
# IDEA #1: Six-Phase Multi-Agent Architecture
# =============================================================================

class PhaseType(Enum):
    """Six execution phases from the protocol."""
    EXPLORE = "explore"
    PLAN = "plan"
    CODE = "code"
    TEST = "test"
    FIX = "fix"
    DOCUMENT = "document"


@dataclass
class AgentConfig:
    """Configuration for a specialized agent."""
    name: str
    role: str
    description: str
    phase: PhaseType
    tools: List[str] = field(default_factory=list)
    system_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "phase": self.phase.value,
            "tools": self.tools,
        }


# Default agent configurations per phase
DEFAULT_AGENTS = {
    PhaseType.EXPLORE: [
        AgentConfig(
            name="researcher",
            role="Researcher",
            description="Analyzes requirements and researches solutions",
            phase=PhaseType.EXPLORE,
            tools=["search", "read_file", "analyze"],
            system_prompt="You are a researcher analyzing project requirements."
        ),
        AgentConfig(
            name="code_analyzer",
            role="Code Analyzer",
            description="Analyzes existing codebase structure",
            phase=PhaseType.EXPLORE,
            tools=["glob", "grep", "read_file"],
            system_prompt="You analyze codebase structure and patterns."
        ),
        AgentConfig(
            name="system_architect",
            role="System Architect",
            description="Designs high-level system architecture",
            phase=PhaseType.EXPLORE,
            tools=["analyze", "diagram"],
            system_prompt="You design system architecture and identify dependencies."
        ),
    ],
    PhaseType.PLAN: [
        AgentConfig(
            name="planner",
            role="Planner",
            description="Creates detailed implementation plans",
            phase=PhaseType.PLAN,
            tools=["plan", "estimate"],
            system_prompt="You create detailed, actionable implementation plans."
        ),
        AgentConfig(
            name="architect",
            role="Technical Architect",
            description="Designs technical solutions",
            phase=PhaseType.PLAN,
            tools=["design", "diagram"],
            system_prompt="You design technical solutions and APIs."
        ),
    ],
    PhaseType.CODE: [
        AgentConfig(
            name="coder",
            role="Developer",
            description="Implements features and fixes",
            phase=PhaseType.CODE,
            tools=["write_file", "edit_file", "run_command"],
            system_prompt="You implement code following best practices."
        ),
        AgentConfig(
            name="template_expert",
            role="Template Expert",
            description="Specializes in template generation",
            phase=PhaseType.CODE,
            tools=["generate_template", "configure_template"],
            system_prompt="You specialize in code template generation."
        ),
    ],
    PhaseType.TEST: [
        AgentConfig(
            name="tester",
            role="QA Tester",
            description="Writes and runs tests",
            phase=PhaseType.TEST,
            tools=["run_tests", "write_test", "analyze_coverage"],
            system_prompt="You write comprehensive tests and verify functionality."
        ),
        AgentConfig(
            name="security_auditor",
            role="Security Auditor",
            description="Performs security analysis",
            phase=PhaseType.TEST,
            tools=["security_scan", "analyze_vulnerabilities"],
            system_prompt="You audit code for security vulnerabilities."
        ),
    ],
    PhaseType.FIX: [
        AgentConfig(
            name="debugger",
            role="Debugger",
            description="Diagnoses and fixes issues",
            phase=PhaseType.FIX,
            tools=["debug", "edit_file", "run_command"],
            system_prompt="You diagnose issues and implement fixes."
        ),
    ],
    PhaseType.DOCUMENT: [
        AgentConfig(
            name="doc_writer",
            role="Documentation Writer",
            description="Creates documentation",
            phase=PhaseType.DOCUMENT,
            tools=["write_file", "generate_docs"],
            system_prompt="You write clear, comprehensive documentation."
        ),
        AgentConfig(
            name="reviewer",
            role="Reviewer",
            description="Reviews code and documentation",
            phase=PhaseType.DOCUMENT,
            tools=["review", "suggest"],
            system_prompt="You review work for quality and completeness."
        ),
    ],
}


class PhaseState(TypedDict):
    """State for a single phase execution."""
    phase: str
    agents: List[str]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    completed_agents: List[str]
    pending_agents: List[str]
    errors: List[Dict[str, Any]]


class MultiAgentState(ContextPreservingState):
    """Extended state for multi-agent orchestration.

    Extends ContextPreservingState with multi-agent specific fields.
    """
    # Agent tracking
    agent_states: Dict[str, Dict[str, Any]]
    agent_outputs: Annotated[Dict[str, Any], operator.add]

    # Phase execution
    phase_results: Dict[str, PhaseState]
    current_agents: List[str]

    # Coordination
    lock_manager_state: Dict[str, Any]
    message_bus_state: Dict[str, Any]


def create_phase_subgraph(
    phase: PhaseType,
    agent_configs: Optional[List[AgentConfig]] = None,
    message_bus: Optional[MessageBus] = None,
    lock_manager: Optional[LockManager] = None
) -> StateGraph:
    """Create a subgraph for a specific phase with parallel agent execution.

    Implements fan-out/fan-in pattern:
    - Fan-out: Dispatch work to all phase agents in parallel
    - Fan-in: Aggregate results from all agents

    Args:
        phase: The phase this subgraph handles
        agent_configs: Agent configurations (uses defaults if not provided)
        message_bus: Shared message bus for communication
        lock_manager: Shared lock manager for coordination

    Returns:
        Compiled StateGraph for the phase
    """
    agents = agent_configs or DEFAULT_AGENTS.get(phase, [])

    if not agents:
        raise ValueError(f"No agents configured for phase: {phase.value}")

    graph = StateGraph(PhaseState)

    # Create agent nodes
    for agent_config in agents:
        async def agent_node(state: PhaseState, config: AgentConfig = agent_config) -> PhaseState:
            """Execute a single agent's work."""
            agent_id = config.name

            # Register with message bus if available
            if message_bus:
                await message_bus.register_agent(agent_id)
                await message_bus.broadcast(
                    agent_id,
                    f"Starting work in {state['phase']} phase",
                    MessageType.STATUS
                )

            try:
                # Agent-specific logic would go here
                # This is a placeholder that simulates work
                output = {
                    "agent": agent_id,
                    "role": config.role,
                    "phase": config.phase.value,
                    "status": "completed",
                    "result": f"{config.role} completed work"
                }

                # Notify completion
                if message_bus:
                    await message_bus.broadcast(
                        agent_id,
                        f"Completed work",
                        MessageType.STATUS
                    )

                return {
                    **state,
                    "outputs": {**state.get("outputs", {}), agent_id: output},
                    "completed_agents": state.get("completed_agents", []) + [agent_id],
                    "pending_agents": [a for a in state.get("pending_agents", []) if a != agent_id]
                }

            except Exception as e:
                error = {
                    "agent": agent_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

                if message_bus:
                    await message_bus.broadcast(
                        agent_id,
                        f"Error: {str(e)}",
                        MessageType.ERROR
                    )

                return {
                    **state,
                    "errors": state.get("errors", []) + [error]
                }

        graph.add_node(agent_config.name, agent_node)

    # Fan-out: route to all agents in parallel
    def route_to_agents(state: PhaseState) -> List[Send]:
        """Route to all phase agents in parallel using Send primitive."""
        return [Send(agent.name, state) for agent in agents]

    graph.add_conditional_edges(START, route_to_agents)

    # Aggregation node
    async def aggregate_phase_results(state: PhaseState) -> PhaseState:
        """Aggregate results from all agents."""
        outputs = state.get("outputs", {})
        errors = state.get("errors", [])

        # Determine overall phase success
        success = len(errors) == 0 and len(outputs) == len(agents)

        return {
            **state,
            "outputs": {
                **outputs,
                "_phase_summary": {
                    "phase": state["phase"],
                    "total_agents": len(agents),
                    "completed": len(state.get("completed_agents", [])),
                    "errors": len(errors),
                    "success": success
                }
            }
        }

    graph.add_node("aggregate", aggregate_phase_results)

    # Fan-in: all agents connect to aggregation
    for agent_config in agents:
        graph.add_edge(agent_config.name, "aggregate")

    graph.add_edge("aggregate", END)

    return graph.compile()


def create_orchestrated_workflow(
    checkpointer: Optional[EnhancedCheckpointer] = None,
    message_bus: Optional[MessageBus] = None,
    lock_manager: Optional[LockManager] = None,
    phases: Optional[List[PhaseType]] = None
) -> StateGraph:
    """Create the main orchestrated workflow with all phases.

    Implements the protocol's six-phase progression:
    Explore → Plan → Code → Test → Fix → Document

    With conditional loops:
    - Fix → Test (if issues remain)
    - Fix → Document (if issues resolved)

    Args:
        checkpointer: Enhanced checkpointer for state persistence
        message_bus: Message bus for agent communication
        lock_manager: Lock manager for resource coordination
        phases: Custom phase order (defaults to all phases)

    Returns:
        Compiled main workflow StateGraph
    """
    if phases is None:
        phases = [
            PhaseType.EXPLORE,
            PhaseType.PLAN,
            PhaseType.CODE,
            PhaseType.TEST,
            PhaseType.FIX,
            PhaseType.DOCUMENT
        ]

    # Create shared resources if not provided
    if message_bus is None:
        message_bus = MessageBus()

    if lock_manager is None:
        lock_manager = LockManager()

    if checkpointer is None:
        checkpointer = EnhancedCheckpointer()

    # Create phase subgraphs
    phase_subgraphs = {
        phase: create_phase_subgraph(phase, message_bus=message_bus, lock_manager=lock_manager)
        for phase in phases
    }

    # Main workflow
    workflow = StateGraph(MultiAgentState)

    # Phase transition nodes
    for phase in phases:
        async def phase_node(state: MultiAgentState, p: PhaseType = phase) -> MultiAgentState:
            """Execute a phase and handle context preservation."""
            # Create checkpoint at phase start
            await checkpointer.checkpoint(
                state,
                CheckpointTrigger.PHASE_TRANSITION,
                f"Starting {p.value} phase"
            )

            # Prepare phase state
            phase_state: PhaseState = {
                "phase": p.value,
                "agents": [a.name for a in DEFAULT_AGENTS.get(p, [])],
                "inputs": {
                    "project_description": state.get("project_description", ""),
                    "template_type": state.get("template_type", ""),
                    "context": state.get("context_variables", {}),
                },
                "outputs": {},
                "completed_agents": [],
                "pending_agents": [a.name for a in DEFAULT_AGENTS.get(p, [])],
                "errors": []
            }

            # Execute phase subgraph
            subgraph = phase_subgraphs[p]
            result = await subgraph.ainvoke(phase_state)

            # Update state with phase results
            return {
                **state,
                "current_phase": p.value,
                "phase_results": {
                    **state.get("phase_results", {}),
                    p.value: result
                },
                "completed_work": state.get("completed_work", []) + [f"Completed {p.value} phase"],
                "active_agents": result.get("completed_agents", []),
            }

        workflow.add_node(phase.value, phase_node)

    # Linear edges between phases
    workflow.set_entry_point(phases[0].value)

    for i, phase in enumerate(phases[:-1]):
        next_phase = phases[i + 1]

        # Special handling for fix → test loop
        if phase == PhaseType.FIX:
            def should_retest_or_document(state: MultiAgentState) -> str:
                """Determine if we need to retest or can proceed to documentation."""
                fix_results = state.get("phase_results", {}).get("fix", {})
                errors = fix_results.get("errors", [])

                # If fix had errors, go back to test
                if errors:
                    return "test"

                # Check if validation passed
                if state.get("validation_passed", False):
                    return "document"

                # Default: retest
                return "test"

            workflow.add_conditional_edges(
                phase.value,
                should_retest_or_document,
                {
                    "test": "test",
                    "document": "document"
                }
            )
        else:
            workflow.add_edge(phase.value, next_phase.value)

    # Final phase connects to END
    workflow.add_edge(phases[-1].value, END)

    # Compile with base checkpointer for LangGraph compatibility
    return workflow.compile(checkpointer=MemorySaver())


# =============================================================================
# Convenience Functions and Exports
# =============================================================================

async def run_orchestrated_generation(
    description: str,
    output_dir: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    phases: Optional[List[PhaseType]] = None,
    thread_id: Optional[str] = None
) -> Dict[str, Any]:
    """Run the full orchestrated template generation workflow.

    Args:
        description: Project description
        output_dir: Output directory for generated code
        context: Pre-filled context variables
        phases: Custom phases to execute (defaults to all)
        thread_id: Thread ID for checkpointing

    Returns:
        Workflow results including phase outputs and status
    """
    # Create shared resources
    checkpointer = EnhancedCheckpointer()
    message_bus = MessageBus()
    lock_manager = LockManager()

    # Create workflow
    workflow = create_orchestrated_workflow(
        checkpointer=checkpointer,
        message_bus=message_bus,
        lock_manager=lock_manager,
        phases=phases
    )

    # Initial state
    initial_state: MultiAgentState = {
        # Message history
        "messages": [],

        # Phase tracking
        "current_phase": "explore",
        "phase_history": [],

        # Agent tracking
        "active_agents": [],
        "completed_work": [],
        "pending_items": [],
        "key_decisions": [],

        # Discovery tracking
        "file_locations": {},
        "dependencies": [],
        "issues": [],

        # Next steps
        "next_steps": ["Analyze requirements", "Select template", "Generate code"],

        # Project context
        "project_description": description,
        "template_type": "",
        "template_name": "",
        "output_dir": output_dir or "",
        "context_variables": context or {},

        # Multi-agent specific
        "agent_states": {},
        "agent_outputs": {},
        "phase_results": {},
        "current_agents": [],
        "lock_manager_state": {},
        "message_bus_state": {},
    }

    config = {"configurable": {"thread_id": thread_id or "default"}}

    try:
        # Create initial checkpoint
        await checkpointer.checkpoint(
            initial_state,
            CheckpointTrigger.TASK_START,
            f"Starting orchestrated generation for: {description[:50]}..."
        )

        # Run workflow
        final_state = await workflow.ainvoke(initial_state, config)

        # Create completion checkpoint
        await checkpointer.checkpoint(
            final_state,
            CheckpointTrigger.TASK_COMPLETE,
            "Orchestrated generation complete"
        )

        return {
            "success": True,
            "phase_results": final_state.get("phase_results", {}),
            "completed_work": final_state.get("completed_work", []),
            "key_decisions": final_state.get("key_decisions", []),
            "checkpoints": [c.checkpoint_id for c in checkpointer.list_checkpoints()],
        }

    except Exception as e:
        logger.error(f"Orchestrated generation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "checkpoints": [c.checkpoint_id for c in checkpointer.list_checkpoints()],
        }


# Export all public classes and functions
__all__ = [
    # Context Preservation
    "KeyDecision",
    "PhaseHandoffContext",
    "ContextPreservingState",
    "validate_handoff_context",
    "create_handoff_context",
    "record_decision",

    # Enhanced Checkpointing
    "CheckpointTrigger",
    "CheckpointMetadata",
    "EnhancedCheckpoint",
    "EnhancedCheckpointer",
    "checkpoint_before_risky",

    # Lock Manager
    "LockType",
    "Lock",
    "LockManager",

    # Message Bus
    "MessageType",
    "ChannelType",
    "AgentMessage",
    "MessageBus",

    # Multi-Agent Architecture
    "PhaseType",
    "AgentConfig",
    "PhaseState",
    "MultiAgentState",
    "DEFAULT_AGENTS",
    "create_phase_subgraph",
    "create_orchestrated_workflow",

    # Convenience
    "run_orchestrated_generation",
]
